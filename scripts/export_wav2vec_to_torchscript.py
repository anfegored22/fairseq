#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq.models.wav2vec import utils as w2v_utils
from fairseq.models.wav2vec import wav2vec2 as w2v2_mod


def _pad_to_multiple_compat(x, multiple, dim=-1, value=0):
    if x is None:
        return None, 0

    tsz = x.size(dim)

    # Compatible with Python ints and trace-time symbolic/tensor scalar sizes.
    try:
        remainder = int((-tsz) % multiple)
    except TypeError:
        remainder = int(((-tsz) % multiple).item())

    if remainder == 0:
        return x, 0

    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


# Patch wav2vec2 padding helper for torch.jit tracing compatibility.
w2v_utils.pad_to_multiple = _pad_to_multiple_compat
w2v2_mod.pad_to_multiple = _pad_to_multiple_compat


class Wav2VecTorchScriptWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def _lengths_to_padding_mask(
        self, lengths: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        time_ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return time_ids >= lengths.unsqueeze(1)

    def forward(self, source: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        padding_mask = self._lengths_to_padding_mask(lengths, source.size(1))
        transpose_tbc_to_btc = False
        # Disable training-time masking to avoid tracing through NumPy-based
        # mask index generation in wav2vec2 pretraining models.
        try:
            net_output = self.model(
                source=source,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
            )

            if "x" in net_output:
                logits = net_output["x"]
            elif hasattr(self.model, "get_logits"):
                logits = self.model.get_logits(net_output, normalize=False)
                transpose_tbc_to_btc = True
            else:
                logits = net_output["encoder_out"]
                transpose_tbc_to_btc = True
        except TypeError:
            # Fallback path for checkpoints whose forward does not accept
            # mask/features_only kwargs.
            net_output = self.model(source=source, padding_mask=padding_mask)
            if hasattr(self.model, "get_logits"):
                logits = self.model.get_logits(net_output, normalize=False)
                transpose_tbc_to_btc = True
            else:
                logits = net_output["encoder_out"]
                transpose_tbc_to_btc = True

        if isinstance(logits, list):
            logits = logits[0]

        if transpose_tbc_to_btc and logits.dim() == 3:
            logits = logits.transpose(0, 1)

        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a Fairseq wav2vec checkpoint and export a TorchScript model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.ts"),
        help="Output TorchScript path",
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=16000,
        help="Dummy waveform length for tracing (in samples)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size for tracing",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force export on CPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(args.checkpoint)])
    if len(models) != 1:
        raise RuntimeError(f"Expected one model, got {len(models)}")

    base_model = models[0].to(device).eval()
    # Avoid non-scriptable custom autograd op GradMultiply during export.
    for module in base_model.modules():
        if hasattr(module, "feature_grad_mult"):
            module.feature_grad_mult = 1.0

    wrapped = Wav2VecTorchScriptWrapper(base_model).to(device).eval()

    dummy_source = torch.randn(args.batch_size, args.sample_length, device=device)
    dummy_lengths = torch.full(
        (args.batch_size,),
        args.sample_length,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        scripted = torch.jit.trace(wrapped, (dummy_source, dummy_lengths), strict=False)
        scripted = torch.jit.freeze(scripted)

    output_path = args.output
    if output_path.exists() and output_path.is_dir():
        output_path = output_path / "model.ts"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    print(f"Saved TorchScript model to: {output_path}")


if __name__ == "__main__":
    main()
