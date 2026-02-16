# UPS wav2vec2 pretraining

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

This command uploads artifacts and syncs TensorBoard scalar points into MLflow metrics, so plots appear in the MLflow run Metrics tab. Metrics are logged as `train/*` and `valid/*`, while `train_inner/*` is skipped by default to keep curves clean.

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
