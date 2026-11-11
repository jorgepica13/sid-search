# Parches sobre KuaiSearch (base 88a545d)

| Commit | Fichero | Cambio | Motivo |
|---|---|---|---|
| b3d058a | .gitignore | Ignore script outputs (data/, recall/data/*, DSI/, LTRGR/, model/, doc_embedding/) | Keep heavy artifacts out of the fork's tree |
| 6a7f5e4 | .gitignore | Ignore `model_signle/` | DPR `save_dir` in `recall/dpr/config/train_dpr_nq.yaml` was not ignored; checkpoint would leave the submodule with untracked files |
| b12e733 | recall/dpr/train.py | Optional `gradient_checkpointing` (off unless set in the yaml) | Fit batch 512 on 1 GPU to test the in-batch-negatives hypothesis for DPR-SDE (emulates 8×64). Same fp32 math: loss identical, grad rel diff 5e-8 |
