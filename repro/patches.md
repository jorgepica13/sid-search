# Parches sobre KuaiSearch (base 88a545d)

| Commit | Fichero | Cambio | Motivo |
|---|---|---|---|
| b3d058a | .gitignore | Ignore script outputs (data/, recall/data/*, DSI/, LTRGR/, model/, doc_embedding/) | Keep heavy artifacts out of the fork's tree |
| 6a7f5e4 | .gitignore | Ignore `model_signle/` | DPR `save_dir` in `recall/dpr/config/train_dpr_nq.yaml` was not ignored; checkpoint would leave the submodule with untracked files |
| b12e733 | recall/dpr/train.py | Optional `gradient_checkpointing` (off unless set in the yaml) | Fit batch 512 on 1 GPU to test the in-batch-negatives hypothesis for DPR-SDE (emulates 8×64). Same fp32 math: loss identical, grad rel diff 5e-8 |
| 6e020a9 | recall/GR/train.py | `--gradient_accumulation_steps` CLI arg (default 1) instead of hard-coded 1 | Reproduce `torchrun --nproc_per_node=8` × 128 on 1 GPU as 128 × 8 accumulation steps. Checked: MT5 has `model_accepts_loss_kwargs=False`, so Trainer divides the loss by the GA steps (mean of 8 micro-batch means = DDP average) |
