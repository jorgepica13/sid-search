# Resultados de reproducción (KuaiSearch-Lite, recall)

Criterio de reproducido: ±5% relativo en R@10.

| Método | R@10 paper | R@10 nuestro | Δ | HR@10 paper | HR@10 nuestro | Δ |
|---|---|---|---|---|---|---|
| BM25 | 0.0706 | 0.0686 | −2.8% ✓ | 0.1001 | 0.0985 | −1.6% |
| DPR-SDE | 0.0826 | | | 0.1210 | | |
| DSI | 0.0623 | | | 0.0965 | | |
| LTRGR | 0.0688 | | | 0.1049 | | |

## Notas

- **BM25** (`logs/20261111-000315_bm25.out`, sid-search `f0c341b`, KuaiSearch `b3d058a`; auditoría `repro/audit/08_bm25.txt`).
  R@20 0.1014 (−2.3%), HR@20 0.1402 (−1.8%), R@50 0.1509 (−3.5%), HR@50 0.2030 (−2.8%).
  Todas las métricas quedan entre un 1.6% y un 3.5% por debajo del paper.
  El 61% de las queries tiene un empate de score entre los rangos 10 y 11; solo el orden de desempate mueve
  R@10 entre 0.0625 y 0.0768 (medido dentro del top-100), un margen más ancho que el criterio de ±5%.
