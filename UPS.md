# UPS wav2vec2 pretraining

Set these values in `.env` to enable direct MLflow logging during training (no extra flags needed):

```bash
MLFLOW_TRACKING_URI=file:///Users/andres/ups-challenge/fairseq/mlruns
MLFLOW_EXPERIMENT_NAME=ups-w2v2
MLFLOW_RUN_NAME=w2v2_ups_ps
```

Use this command to run a CPU smoke pretraining job on the UPS manifests and write TensorBoard logs.

```bash
uv run fairseq-hydra-train \
  --config-dir examples/wav2vec/config/pretraining \
  --config-name wav2vec2_base_librispeech \
  task.data=/Users/andres/ups-challenge/fairseq/data/ups_ps/manifests \
  distributed_training.distributed_world_size=1 \
  common.cpu=true \
  common.fp16=false \
  task.max_sample_size=120000 \
  dataset.max_tokens=120000 \
  optimization.update_freq='[1]' \
  optimization.max_update=500 \
  dataset.num_workers=2 \
  checkpoint.save_dir=/Users/andres/ups-challenge/fairseq/checkpoints/w2v2_ups_ps \
  checkpoint.save_interval_updates=100 \
  common.log_interval=20 \
  common.tensorboard_logdir=/Users/andres/ups-challenge/fairseq/tb/w2v2_ups_ps
```

When the MLflow vars above are present in your environment (or `.env`), the same run is logged directly to MLflow with split-aware metrics (`train/*` from `train_inner`, plus `valid/*`).

Open TensorBoard in a second terminal:

```bash
uv run tensorboard --logdir /Users/andres/ups-challenge/fairseq/tb/w2v2_ups_ps --port 6006
```

Track the same run in MLflow by logging TensorBoard files and checkpoints as artifacts:

```bash
uv pip install mlflow
uv pip install tensorboard

# one command (after or during training)
./scripts/log_w2v2_mlflow.sh
```

This command uploads artifacts and syncs TensorBoard scalar points into MLflow metrics, so plots appear in the MLflow run Metrics tab. It maps TensorBoard `train_inner/*` to MLflow `train/*` (dense step-level curves), keeps `valid/*`, and skips sparse epoch-level `train/*` to avoid duplicate-looking points.

Open MLflow UI:

```bash
uv run mlflow ui --backend-store-uri file:///Users/andres/ups-challenge/fairseq/mlruns --port 5001
```

Then open `http://localhost:5001`.

Use this command for single-GPU training with fp16 and TensorBoard logs:

```bash
uv run fairseq-hydra-train \
  --config-dir examples/wav2vec/config/pretraining \
  --config-name wav2vec2_base_librispeech \
  task.data=/Users/andres/ups-challenge/fairseq/data/ups_ps/manifests \
  distributed_training.distributed_world_size=1 \
  common.cpu=false \
  common.fp16=true \
  task.max_sample_size=120000 \
  dataset.max_tokens=400000 \
  optimization.update_freq='[4]' \
  optimization.max_update=10000 \
  dataset.num_workers=4 \
  checkpoint.save_dir=/Users/andres/ups-challenge/fairseq/checkpoints/w2v2_ups_ps_gpu \
  checkpoint.save_interval_updates=500 \
  common.log_interval=20 \
  common.tensorboard_logdir=/Users/andres/ups-challenge/fairseq/tb/w2v2_ups_ps_gpu
```
