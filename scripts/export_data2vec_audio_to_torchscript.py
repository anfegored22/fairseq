#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils as fairseq_utils
from fairseq.models.wav2vec import utils as w2v_utils
from fairseq.models.wav2vec import wav2vec2 as w2v2_mod


def _pad_to_multiple_compat(x, multiple, dim=-1, value=0):
    if x is None:
        return None, 0

    tsz = x.size(dim)

    try:
        remainder = int((-tsz) % multiple)
    except TypeError:
        remainder = int(((-tsz) % multiple).item())

    if remainder == 0:
        return x, 0

    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


w2v_utils.pad_to_multiple = _pad_to_multiple_compat  # type: ignore[attr-defined]
w2v2_mod.pad_to_multiple = _pad_to_multiple_compat


class Data2VecAudioTorchScriptWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def _lengths_to_padding_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        time_ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return time_ids >= lengths.unsqueeze(1)

    def _extract_x(self, net_output) -> torch.Tensor:
        if isinstance(net_output, dict):
            if "x" in net_output and isinstance(net_output["x"], torch.Tensor):
                return net_output["x"]
            if "encoder_out" in net_output:
                out = net_output["encoder_out"]
                if isinstance(out, list):
                    out = out[0]
                if isinstance(out, torch.Tensor):
                    if out.dim() == 3:
                        out = out.transpose(0, 1)
                    return out

        if isinstance(net_output, tuple):
            out = net_output[0]
            if isinstance(out, list):
                out = out[0]
            if isinstance(out, torch.Tensor):
                return out

        if isinstance(net_output, torch.Tensor):
            return net_output

        raise RuntimeError("Could not extract Tensor features from model output")

    def forward(self, source: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        padding_mask = self._lengths_to_padding_mask(lengths, source.size(1))

        try:
            net_output = self.model(
                source=source,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
            )
        except TypeError:
            net_output = self.model(source=source, padding_mask=padding_mask)

        x = self._extract_x(net_output)
        if x.dim() == 3 and x.size(0) == source.size(1) and x.size(1) == source.size(0):
            x = x.transpose(0, 1)

        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Fairseq data2vec-audio checkpoint and export a TorchScript model."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to checkpoint (.pt)"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("model.ts"), help="Output TorchScript path"
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=16000,
        help="Dummy waveform length for tracing (in samples)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Dummy batch size for tracing"
    )
    parser.add_argument("--cpu", action="store_true", help="Force export on CPU")
    parser.add_argument(
        "--user-dir",
        type=Path,
        default=Path("examples/data2vec"),
        help="User directory to import model/task registrations from",
    )
    return parser.parse_args()


def _import_data2vec_registrations(user_dir: Path) -> None:
    try:
        fairseq_utils.import_user_module(argparse.Namespace(user_dir=str(user_dir)))
        return
    except ModuleNotFoundError as exc:
        model_file = user_dir / "models" / "data2vec_audio.py"
        if not model_file.exists():
            raise RuntimeError(
                f"Failed to import --user-dir={user_dir} and fallback file was not found: {model_file}"
            ) from exc

        spec = importlib.util.spec_from_file_location(
            "export_data2vec_audio_user_model", str(model_file)
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not create module spec for: {model_file}"
            ) from exc

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(
            "Warning: full --user-dir import failed with missing dependency "
            f"({exc}). Loaded model registration from {model_file} only."
        )


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.user_dir is not None:
        _import_data2vec_registrations(args.user_dir)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.checkpoint)], strict=False
    )
    if len(models) != 1:
        raise RuntimeError(f"Expected one model, got {len(models)}")

    base_model = models[0].to(device).eval()
    for module in base_model.modules():
        if hasattr(module, "feature_grad_mult"):
            module.feature_grad_mult = 1.0

    wrapped = Data2VecAudioTorchScriptWrapper(base_model).to(device).eval()

    dummy_source = torch.randn(args.batch_size, args.sample_length, device=device)
    dummy_lengths = torch.full(
        (args.batch_size,),
        args.sample_length,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        wrapped(dummy_source, dummy_lengths)
        scripted = torch.jit.trace(
            wrapped,
            (dummy_source, dummy_lengths),
            strict=False,
            check_trace=False,
        )
        scripted = torch.jit.freeze(scripted)

    output_path = args.output
    if output_path.exists() and output_path.is_dir():
        output_path = output_path / "model.ts"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    print(f"Saved TorchScript model to: {output_path}")


if __name__ == "__main__":
    main()
