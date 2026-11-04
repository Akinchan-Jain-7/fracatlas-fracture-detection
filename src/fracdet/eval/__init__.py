"""Phase 3: choosing the operating threshold on validation and the one-shot test evaluation.

- ``metrics``    confusion counts and the full metric set at one threshold
- ``threshold``  every operating point on a score set; the pre-registered rule
- ``bootstrap``  cluster-bootstrap confidence intervals
- ``report``     subgroup tables (all / no hardware / per region)
- ``protocol``   pre-registration and "test only once" guards
- ``predict``    rebuild a model + preprocessing from a checkpoint and score a split
- ``plots``      PR / ROC curves, threshold sweep, confusion matrix
"""
