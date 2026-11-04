"""Guards that make "the test set is scored once, with a pre-chosen threshold" checkable.

Promises in a README are easy to break by accident: tweak the threshold after
seeing a disappointing test score, re-run with "one small fix", repeat. These
checks turn the promises into errors:

1. **Pre-registration.** The threshold file must be committed in git, unmodified,
   before the test set is scored. Git history then proves the threshold existed
   before the test result did.
2. **Same model.** The checkpoint must be the one the threshold was tuned for
   (same best epoch, training commit, split file), and re-scoring validation
   with it must reproduce the validation numbers in the threshold file.
3. **Once.** If test results are already committed for this model, a re-run
   needs an explicit, recorded reason (for example "evaluation code crashed
   before writing results").
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class ProtocolError(RuntimeError):
    """The evaluation would break the test-set protocol."""


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def committed_and_unchanged(path: str | Path) -> tuple[bool, str]:
    """True if ``path`` is tracked by git and identical to the committed version."""
    path = Path(path).resolve()
    if not path.is_file():
        return False, f"{path} does not exist"
    cwd = path.parent
    if _git(["ls-files", "--error-unmatch", path.name], cwd).returncode != 0:
        return False, f"{path.name} is not committed to git (untracked)"
    if _git(["diff", "--quiet", "HEAD", "--", path.name], cwd).returncode != 0:
        return False, f"{path.name} has uncommitted changes"
    commit = _git(["log", "-1", "--format=%H %cI", "--", path.name], cwd).stdout.strip()
    return True, commit


def check_preregistered(threshold_path: str | Path) -> dict[str, Any]:
    ok, info = committed_and_unchanged(threshold_path)
    if not ok:
        raise ProtocolError(
            f"Threshold not pre-registered: {info}. Commit and push the threshold file "
            "before scoring the test set (see docs/phase-3-evaluation.md)."
        )
    commit, _, committed_at = info.partition(" ")
    return {"threshold_commit": commit, "threshold_committed_at": committed_at}


def check_not_already_evaluated(results_dir: str | Path, rerun_reason: str | None) -> None:
    done = Path(results_dir) / "test_metrics.json"
    if done.exists() and not rerun_reason:
        raise ProtocolError(
            f"{done} already exists: this model's test set has been scored. "
            "Re-running needs --rerun-reason '<why>', which is written into the results."
        )


def check_same_model(ckpt: dict[str, Any], record: dict[str, Any], split_md5: str) -> None:
    """The checkpoint, the threshold record and the split file must all agree."""
    model = record["model"]
    problems = []
    if ckpt["epoch"] != model["best_epoch"]:
        problems.append(
            f"checkpoint epoch {ckpt['epoch']} != tuned-for epoch {model['best_epoch']}"
        )
    trained = ckpt["provenance"].get("git", {}).get("commit")
    if trained != model["trained_commit"]:
        problems.append(
            f"checkpoint commit {trained} != tuned-for commit {model['trained_commit']}"
        )
    for name, value in [
        ("checkpoint", ckpt["provenance"].get("split_csv_md5")),
        ("threshold file", model["split_csv_md5"]),
    ]:
        if value != split_md5:
            problems.append(f"{name} split MD5 {value} != current split file {split_md5}")
    if problems:
        raise ProtocolError("Model does not match the threshold record: " + "; ".join(problems))


def check_val_reproduced(
    rescored: dict[str, Any],
    record: dict[str, Any],
    pr_auc_tol: float = 2e-3,
    max_flips: int = 2,
) -> dict[str, Any]:
    """Validation re-scored now vs. when the threshold was chosen.

    Small differences are expected (mixed precision on a different GPU), so the
    check allows a PR-AUC change of ``pr_auc_tol`` and up to ``max_flips``
    images changing side of the threshold. More than that means a different
    model or different preprocessing, and the threshold would not transfer.
    """
    saved = record["val"]
    d_pr = abs(rescored["pr_auc"] - saved["pr_auc"])
    flips = abs(rescored["tp"] - saved["at_threshold"]["tp"]) + abs(
        rescored["fp"] - saved["at_threshold"]["fp"]
    )
    report = {"pr_auc_saved": saved["pr_auc"], "pr_auc_now": rescored["pr_auc"], "flips": flips}
    if d_pr > pr_auc_tol or flips > max_flips:
        raise ProtocolError(f"Validation not reproduced by this checkpoint: {report}")
    return report
