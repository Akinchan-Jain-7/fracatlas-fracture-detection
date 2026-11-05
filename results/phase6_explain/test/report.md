## test: Grad-CAM localisation (fractured images)

Pointing game = share of fractured images whose heatmap peak is inside a fracture box. 95% CIs resample patient groups.

| Group | n | hit rate, stages.3 | hit rate, stages.2 | centre | chance | energy share / area share, stages.3 |
|---|---|---|---|---|---|---|
| all | 102 | 0.225 [0.135, 0.318] | 0.353 [0.233, 0.472] | 0.078 [0.010, 0.157] | 0.010 | 0.078 / 0.010 |
| outcome=FN | 8 | 0.000 [0.000, 0.000] | 0.125 [0.000, 0.333] | 0.125 [0.000, 0.333] | 0.006 | 0.014 / 0.006 |
| outcome=TP | 94 | 0.245 [0.147, 0.351] | 0.372 [0.241, 0.510] | 0.074 [0.011, 0.150] | 0.010 | 0.083 / 0.010 |
| region=hand | 54 | 0.185 [0.074, 0.320] | 0.333 [0.180, 0.500] | 0.000 [0.000, 0.000] | 0.005 | 0.050 / 0.005 |
| region=hip | 1 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.032 | 0.393 / 0.032 |
| region=leg | 30 | 0.267 [0.095, 0.405] | 0.433 [0.185, 0.636] | 0.233 [0.000, 0.423] | 0.014 | 0.110 / 0.014 |
| region=mixed | 17 | 0.235 [0.000, 0.533] | 0.235 [0.000, 0.533] | 0.059 [0.000, 0.200] | 0.014 | 0.089 / 0.014 |

Outcomes at threshold 0.0968: {'tp': 94, 'fp': 142, 'tn': 345, 'fn': 8} (committed: {'tp': 94, 'fp': 142, 'tn': 345, 'fn': 8}). Plan committed: True (0e52ac7ea45ceb3ff5b3e38c6d94165224637735 2026-11-05T07:12:28+00:00). Device: Tesla T4, 58.7 s.
