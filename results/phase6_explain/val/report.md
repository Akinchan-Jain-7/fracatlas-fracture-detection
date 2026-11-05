## val: Grad-CAM localisation (fractured images)

Pointing game = share of fractured images whose heatmap peak is inside a fracture box. 95% CIs resample patient groups.

| Group | n | hit rate, stages.3 | hit rate, stages.2 | centre | chance | energy share / area share, stages.3 |
|---|---|---|---|---|---|---|
| all | 102 | 0.216 [0.126, 0.325] | 0.353 [0.255, 0.449] | 0.049 [0.011, 0.094] | 0.008 | 0.070 / 0.008 |
| outcome=FN | 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.200 [0.000, 0.500] | 0.005 | 0.008 / 0.005 |
| outcome=TP | 92 | 0.239 [0.141, 0.352] | 0.391 [0.290, 0.495] | 0.033 [0.000, 0.069] | 0.009 | 0.076 / 0.009 |
| region=hand | 54 | 0.167 [0.078, 0.260] | 0.444 [0.319, 0.594] | 0.019 [0.000, 0.050] | 0.005 | 0.044 / 0.005 |
| region=hip | 2 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.015 | 0.041 / 0.015 |
| region=leg | 31 | 0.258 [0.037, 0.500] | 0.226 [0.074, 0.359] | 0.097 [0.000, 0.200] | 0.012 | 0.101 / 0.012 |
| region=mixed | 15 | 0.333 [0.125, 0.571] | 0.333 [0.067, 0.643] | 0.067 [0.000, 0.200] | 0.012 | 0.099 / 0.012 |

## Sanity check: randomised weights (64 fractured validation images)

Trained model hit rate on these images: 0.219

| Randomised | Spearman with trained maps (median) | hit rate | flat maps |
|---|---|---|---|
| head_and_last_stage | 0.350 | 0.125 | 0 |
| full | 0.136 | 0.031 | 0 |

Outcomes at threshold 0.0968: {'tp': 92, 'fp': 131, 'tn': 353, 'fn': 10} (committed: {'tp': 92, 'fp': 131, 'tn': 353, 'fn': 10}). Plan committed: True (0e52ac7ea45ceb3ff5b3e38c6d94165224637735 2026-11-05T07:12:28+00:00). Device: Tesla T4, 68.4 s.
