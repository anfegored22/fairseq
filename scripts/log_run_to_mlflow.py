#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable

import mlflow


def parse_tags(values: list[str]) -> dict[str, str]:
    tags: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid tag '{item}'. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid tag '{item}'. Empty key")
        tags[key] = value
    return tags


def ensure_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def iter_event_files(tb_dir: Path) -> Iterable[Path]:
    yield from tb_dir.rglob("events.out.tfevents.*")


def sync_tensorboard_scalars_to_mlflow(tb_dir: Path) -> tuple[int, int]:
    from tensorboard.backend.event_processing import event_accumulator

    num_files = 0
    num_points = 0
    logged_by_metric_step: dict[tuple[str, int], tuple[int, float]] = {}

    for event_file in iter_event_files(tb_dir):
        split_name = event_file.parent.name
        if split_name == "train_inner":
            continue

        num_files += 1
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,
            },
        )
        accumulator.Reload()
        scalar_tags = accumulator.Tags().get("scalars", [])
        for tag in scalar_tags:
            events = accumulator.Scalars(tag)
            events = sorted(events, key=lambda e: (e.step, e.wall_time))
            for ev in events:
                metric_name = f"{split_name}/{tag}"
                step = int(ev.step)
                timestamp_ms = int(ev.wall_time * 1000)
                logged_by_metric_step[(metric_name, step)] = (
                    timestamp_ms,
                    float(ev.value),
                )

    for (metric_name, step), (timestamp_ms, value) in sorted(
        logged_by_metric_step.items(), key=lambda x: (x[0][0], x[0][1], x[1][0])
    ):
        mlflow.log_metric(metric_name, value, step=step, timestamp=timestamp_ms)
        num_points += 1

    return num_files, num_points


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log TensorBoard and checkpoint directories to MLflow artifacts"
    )
    parser.add_argument(
        "--tb-dir",
        required=True,
        help="TensorBoard log directory to upload as artifacts",
    )
    parser.add_argument(
        "--ckpt-dir",
        required=True,
        help="Checkpoint directory to upload as artifacts",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: file://<repo>/mlruns)",
    )
    parser.add_argument(
        "--experiment",
        default="ups-w2v2",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        default="w2v2_ups_ps",
        help="MLflow run name",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Run tag in KEY=VALUE format (repeatable)",
    )
    parser.add_argument(
        "--no-sync-scalars",
        action="store_true",
        help="Do not copy TensorBoard scalar points into MLflow metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()

    repo_root = Path(__file__).resolve().parents[1]
    tb_dir = Path(args.tb_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    ensure_dir(tb_dir, "TensorBoard directory")
    ensure_dir(ckpt_dir, "Checkpoint directory")

    tracking_uri = args.tracking_uri
    if tracking_uri is None:
        tracking_uri = f"file://{(repo_root / 'mlruns').resolve()}"

    tags = parse_tags(args.tag)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_param("tensorboard_dir", str(tb_dir))
        mlflow.log_param("checkpoint_dir", str(ckpt_dir))
        mlflow.log_artifacts(str(tb_dir), artifact_path="tensorboard")
        mlflow.log_artifacts(str(ckpt_dir), artifact_path="checkpoints")

        if not args.no_sync_scalars:
            try:
                num_files, num_points = sync_tensorboard_scalars_to_mlflow(tb_dir)
                mlflow.log_param("tb_event_files_parsed", num_files)
                mlflow.log_param("tb_scalar_points_logged", num_points)
                print(
                    f"Synced TensorBoard scalars: files={num_files}, points={num_points}"
                )
            except ImportError:
                print(
                    "TensorBoard scalars were not synced (tensorboard package not installed). "
                    "Install with: uv pip install tensorboard"
                )

    print("Logged artifacts to MLflow")
    print(f"tracking_uri={tracking_uri}")
    print(f"experiment={args.experiment}")
    print(f"run_name={args.run_name}")


if __name__ == "__main__":
    main()
