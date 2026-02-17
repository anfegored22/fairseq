#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from fairseq import checkpoint_utils


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
        net_output = self.model(source=source, padding_mask=padding_mask)

        if hasattr(self.model, "get_logits"):
            logits = self.model.get_logits(net_output, normalize=False)
        else:
            logits = net_output["encoder_out"]

        if isinstance(logits, list):
            logits = logits[0]

        if logits.dim() == 3 and logits.size(1) == source.size(0):
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(args.output))

    print(f"Saved TorchScript model to: {args.output}")


if __name__ == "__main__":
    main()
