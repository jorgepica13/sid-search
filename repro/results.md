# Resultados de reproducción (KuaiSearch-Lite, recall)

Criterio de reproducido: ±5% relativo en R@10.

| Método | R@10 paper | R@10 nuestro | Δ | HR@10 paper | HR@10 nuestro | Δ |
|---|---|---|---|---|---|---|
| BM25 | 0.0706 | 0.0686 | −2.8% ✓ | 0.1001 | 0.0985 | −1.6% |
| DPR-SDE | 0.0826 | 0.0852 | +3.1% ✓ | 0.1210 | 0.1240 | +2.5% |
| DSI | 0.0623 | 0.0634 | +1.7% ✓ | 0.0965 | 0.0979 | +1.4% |
| LTRGR | 0.0688 | 0.0678 | −1.5% ✓ | 0.1049 | 0.1042 | −0.7% |

## Notas

- **BM25** (`logs/20261111-000315_bm25.out`, sid-search `f0c341b`, KuaiSearch `b3d058a`; auditoría `repro/audit/08_bm25.txt`).
  R@20 0.1014 (−2.3%), HR@20 0.1402 (−1.8%), R@50 0.1509 (−3.5%), HR@50 0.2030 (−2.8%).
  Todas las métricas quedan entre un 1.6% y un 3.5% por debajo del paper.
  El 61% de las queries tiene un empate de score entre los rangos 10 y 11; solo el orden de desempate mueve
  R@10 entre 0.0625 y 0.0768 (medido dentro del top-100), un margen más ancho que el criterio de ±5%.
- **DPR-SDE** (`logs/20261111-001652_dpr_sde.out`, sid-search `c3ae66f`, KuaiSearch `6a7f5e4`; auditoría `repro/audit/09_dpr_sde.txt`).
  1 GPU, batch 64, 1 semilla, early stop en la época 7 (mejor modelo: época 5).
  R@20 0.1153 (−10.8%), HR@20 0.1625 (−10.4%), R@50 0.1840 (−11.5%), HR@50 0.2504 (−9.6%). **No reproduce.**
  Con el orden de empates más favorable R@10 llega a 0.0777 (−5.9%), así que los empates no explican la diferencia.
  Hipótesis principal (deducción): negativos in-batch, 64×128 con 1 GPU frente a 512×1024 si ellos usaron 8 GPUs.
- **DPR-SDE, ejecución 2: batch 512** (`logs/20261111-085916_dpr_sde_bs512.out`, sid-search `c3b6b80`, KuaiSearch `b12e733`, `configs/dpr_sde_bs512.yaml`; auditoría `repro/audit/10_dpr_sde_bs512.txt`).
  Solo cambia el batch (64 → 512, con gradient checkpointing, parche 3, misma matemática). **Reproduce.**
  R@20 0.1297 (+0.3%), HR@20 0.1816 (+0.1%), R@50 0.2080 (+0.05%), HR@50 0.2789 (+0.7%). Early stop en la época 12 (mejor modelo: época 10).
  De la ejecución 1 a la 2, R@10 pasa de 0.0753 a 0.0852: los resultados del paper son compatibles con un batch total de 512.
  La fila de la tabla es la de la ejecución 2.
- **DSI** (`logs/20261111-154005_dsi.out`, sid-search `68614f4`, KuaiSearch `6e020a9`; auditoría `repro/audit/11_dsi.txt`).
  1 GPU, 128 × 8 pasos de acumulación = batch total 1,024 (parche 4), 30/30 épocas, 35 h 52 min. **Reproduce.**
  R@20 0.0887 (−0.6%), HR@20 0.1364 (+1.5%), R@50 0.1345 (−1.7%), HR@50 0.1998 (−1.0%).
- **LTRGR** (`logs/20261113-084527_ltrgr.out`, sid-search `86e0a54`, KuaiSearch `6e020a9`; auditoría `repro/audit/12_ltrgr.txt`).
  `recall_gr.sh` + `--LTRGR_flag 1` (su repo no trae script para LTRGR) desde `DSI/best_model`, 128 × 8 = 1,024, 5/5 épocas, 6 h 49 min. **Reproduce.**
  R@20 0.0961 (−2.5%), HR@20 0.1463 (−0.9%), R@50 0.1442 (−3.9%), HR@50 0.2113 (−3.3%).
