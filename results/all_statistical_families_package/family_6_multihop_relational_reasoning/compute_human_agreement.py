"""Compute human-human Cohen's kappa for the RAG answer audit.

Two completed reviewer CSVs are required. The script reports observed
agreement, Cohen's kappa, and a seeded bootstrap interval for faithfulness,
relevance, and correctness. If an independent-LLM JSON file is supplied, it
also reports secondary human-vs-model agreement; that secondary value must not
be described as human Cohen's kappa.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REVIEWER_1 = REPO_ROOT / "ai_engine" / "rag" / "data" / "human_audit" / "reviewer_1.csv"
DEFAULT_REVIEWER_2 = REPO_ROOT / "ai_engine" / "rag" / "data" / "human_audit" / "reviewer_2.csv"
DEFAULT_OUTPUT = REPO_ROOT / "ai_engine" / "rag" / "data" / "human_audit" / "human_audit_agreement.json"
DEFAULT_SEED = 20261013
DEFAULT_BOOTSTRAP = 2000

LABELS = {
    "faithfulness_label": ("supported", "mixed", "unsupported"),
    "relevance_label": ("direct", "partial", "off_topic"),
    "correctness_label": ("correct", "uncertain", "incorrect"),
}


def load_csv(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    output: Dict[str, Dict[str, str]] = {}
    for row in rows:
        audit_id = (row.get("audit_id") or "").strip()
        if not audit_id or audit_id in output:
            raise ValueError(f"Missing or duplicate audit_id in {path}: {audit_id!r}")
        output[audit_id] = {key: (value or "").strip() for key, value in row.items()}
    return output


def validate_pair(
    first: Mapping[str, Mapping[str, str]],
    second: Mapping[str, Mapping[str, str]],
) -> List[str]:
    if set(first) != set(second):
        raise ValueError("Reviewer CSVs contain different audit IDs")
    ids = sorted(first)
    for audit_id in ids:
        for field, allowed in LABELS.items():
            for reviewer, row in (("reviewer 1", first[audit_id]), ("reviewer 2", second[audit_id])):
                value = row.get(field, "")
                if value not in allowed:
                    raise ValueError(
                        f"{reviewer} has invalid/missing {field} for {audit_id}: "
                        f"{value!r}; expected one of {allowed}"
                    )
    return ids


def cohen_kappa(first: Sequence[str], second: Sequence[str], categories: Sequence[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Cohen's kappa requires equal, non-empty label arrays")
    category_set = set(categories)
    if any(label not in category_set for label in list(first) + list(second)):
        raise ValueError("Observed label is outside the supplied category set")
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    expected = sum(
        first.count(category) * second.count(category) for category in categories
    ) / (n * n)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def bootstrap_ci(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str],
    resamples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(first)
    estimates = []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        estimates.append(
            cohen_kappa(
                [first[index] for index in indices],
                [second[index] for index in indices],
                categories,
            )
        )
    estimates.sort()
    low_index = max(0, int(0.025 * resamples) - 1)
    high_index = min(resamples - 1, int(0.975 * resamples))
    return estimates[low_index], estimates[high_index]


def agreement_summary(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str],
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    kappa = cohen_kappa(first, second, categories)
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    low, high = bootstrap_ci(first, second, categories, resamples, seed)
    return {
        "n": len(first),
        "observed_agreement": round(observed, 4),
        "cohen_kappa": round(kappa, 4),
        "bootstrap_ci95": [round(low, 4), round(high, 4)],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "categories": list(categories),
    }


def independent_labels(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("labels")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No labels found in {path}")
    output = {}
    for row in rows:
        audit_id = row.get("audit_id")
        if not isinstance(audit_id, str) or audit_id in output:
            raise ValueError(f"Missing or duplicate audit_id in {path}")
        output[audit_id] = row
    return output


def compute_agreement(
    reviewer_1_path: Path,
    reviewer_2_path: Path,
    output_path: Path,
    independent_path: Path | None,
    resamples: int,
    seed: int,
) -> None:
    reviewer_1 = load_csv(reviewer_1_path)
    reviewer_2 = load_csv(reviewer_2_path)
    audit_ids = validate_pair(reviewer_1, reviewer_2)
    output: Dict[str, Any] = {
        "audit_type": "human_human_agreement",
        "human_reviewers": 2,
        "n": len(audit_ids),
        "status": "completed",
        "dimensions": {},
    }
    for offset, (field, categories) in enumerate(LABELS.items()):
        first = [reviewer_1[audit_id][field] for audit_id in audit_ids]
        second = [reviewer_2[audit_id][field] for audit_id in audit_ids]
        output["dimensions"][field] = agreement_summary(
            first, second, categories, resamples, seed + offset
        )

    if independent_path:
        independent = independent_labels(independent_path)
        if set(independent) != set(audit_ids):
            raise ValueError("Independent judge and human files contain different audit IDs")
        output["human_vs_independent_judge"] = {}
        for offset, (field, categories) in enumerate(LABELS.items()):
            first = [reviewer_1[audit_id][field] for audit_id in audit_ids]
            second = [str(independent[audit_id].get(field, "")) for audit_id in audit_ids]
            if any(value not in categories for value in second):
                raise ValueError(f"Independent judge has invalid {field} labels")
            output["human_vs_independent_judge"][field] = agreement_summary(
                first, second, categories, resamples, seed + 100 + offset
            )
        output["human_vs_independent_judge_note"] = (
            "Secondary human-versus-model agreement; this is not human Cohen's kappa."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote agreement results to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-1", type=Path, default=DEFAULT_REVIEWER_1)
    parser.add_argument("--reviewer-2", type=Path, default=DEFAULT_REVIEWER_2)
    parser.add_argument("--independent-judge", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    compute_agreement(
        args.reviewer_1,
        args.reviewer_2,
        args.output,
        args.independent_judge,
        args.bootstrap_resamples,
        args.seed,
    )


if __name__ == "__main__":
    main()
