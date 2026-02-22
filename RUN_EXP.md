## Curriculum masking
uv run fairseq-hydra-train \
  --config-dir examples/data2vec/config/v2 \
  --config-name ups_audio_curriculum_local \
  task.data=/Users/andres/ups-challenge/fairseq/data/ups_ps/manifests \
  model.modalities.audio.mask_cluster_count=0 \
  dataset.train_subset=train \
  dataset.valid_subset=valid \
  checkpoint.finetune_from_model=/Users/andres/Downloads/base_libri.pt\
  common.mlflow_experiment=data2vec2\
  common.mlflow_run_name=d2v2_curriculum_masking\
  common.log_interval=1

## Normal
PYTHONPATH=. uv run --env-file .env fairseq-hydra-train \
  --config-dir examples/data2vec/config/v2 \
  --config-name ups_audio_no_curriculum.yaml\
  task.data=/Users/andres/ups-challenge/fairseq/data/ups_ps/manifests \
  model.modalities.audio.mask_cluster_count=0 \
  dataset.train_subset=train \
  dataset.valid_subset=valid \
  checkpoint.finetune_from_model=/Users/andres/Downloads/base_libri.pt\
  common.mlflow_experiment=data2vec2\
  common.mlflow_run_name=d2v2_curriculum_masking\
  common.log_interval=1

## Loss sampling
PYTHONPATH=. uv run --env-file .env fairseq-hydra-train \
  --config-dir examples/data2vec/config/v2 \
  --config-name ups_audio_curriculum_local \
  task.data=/Users/andres/ups-challenge/fairseq/data/ups_ps/manifests \
  dataset.train_subset=train \
  dataset.valid_subset=valid \
  checkpoint.finetune_from_model=/Users/andres/Downloads/base_libri.pt\
  common.mlflow_experiment=data2vec2\
  common.mlflow_run_name=d2v2_curriculum_masking\
  model.modalities.audio.mask_cluster_loss_ema_decay=0.6 \
  common.log_interval=1


## Reference
Adversarial Masking https://arxiv.org/pdf/2201.13100
