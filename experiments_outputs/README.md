

## model folder:

- `best_model_checkpoint.pt`: the final trained SSM-CGM Stream model. Despite the
  "10epoch" naming on the eval output folders, the config each eval run actually
  loaded (`ckpt:` field) points here: this is a checkpoint from
  `outputs/aireadi_stream_mamba_stateful_5epoch/`, not a separate 10-epoch training run.
- `eval_test_config_resolved.yaml` / `eval_validation_config_resolved.yaml`: the
  resolved configs for the two eval runs that produced the main-report and appendix
  metrics.
- `training_curves.png`: appendix "Training configuration and convergence" figure.



