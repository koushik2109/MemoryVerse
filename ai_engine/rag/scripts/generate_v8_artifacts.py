#!/usr/bin/env python3
"""Generate V8 reconciliation artifacts from frozen V3/V7 result files.

This script only reads existing result files and writes derived audit tables under
results/v8. It does not rerun retrieval, change a split, or create measurements.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = RESULTS / "v8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_dose_response() -> None:
    grid_path = RESULTS / "v3" / "headline_regime_grid_v3.csv"
    engine_path = RESULTS / "v3" / "engine_replication.csv"
    grid = read_csv(grid_path)
    engine = read_csv(engine_path)

    selected: list[dict[str, object]] = []
    # The public regime grid is the authoritative Apple Vision source for the
    # main held-out regime analysis. The engine replication artifact is the
    # authoritative Tesseract source for the cross-engine comparison.
    for row in grid:
        if row["Condition"] == "clean_ocr" and row["System"] in {
            "BM25 (Tuned)",
            "BM25 (Char 3-5gram)",
            "Dense MiniLM (chunked)",
            "Dense BGE-base (chunked)",
            "Tuned Hybrid",
        } and row["Query_Type"] in {"verbatim", "light", "heavy"}:
            selected.append(
                {
                    "engine": "Apple Vision",
                    "source_file": str(grid_path.relative_to(ROOT)),
                    "source_row": grid.index(row) + 2,
                    "condition": row["Condition"],
                    "query_type": row["Query_Type"],
                    "system": row["System"],
                    "mrr": row["MRR"],
                    "ci_lower": row["MRR_CI_Lower"],
                    "ci_upper": row["MRR_CI_Upper"],
                }
            )

    for row in engine:
        if row["Engine"] == "Tesseract 5.5" and row["System"] in {
            "BM25 (Tuned)",
            "BM25 (Char 3-5gram)",
            "Dense MiniLM",
            "Dense BGE-base",
            "Tuned Hybrid",
        } and row["Query_Type"] in {"verbatim", "light", "heavy"} and row["Condition"] in {
            "clean_ocr",
            "degraded_ocr",
        }:
            selected.append(
                {
                    "engine": "Tesseract 5.5",
                    "source_file": str(engine_path.relative_to(ROOT)),
                    "source_row": engine.index(row) + 2,
                    "condition": row["Condition"],
                    "query_type": row["Query_Type"],
                    "system": row["System"],
                    "mrr": row["MRR"],
                    "ci_lower": row["MRR_CI_Lower"],
                    "ci_upper": row["MRR_CI_Upper"],
                }
            )

    lookup = {(r["engine"], r["query_type"], r["condition"], r["system"]): r for r in selected}
    drops: list[dict[str, object]] = []
    for engine_name in ("Apple Vision", "Tesseract 5.5"):
        condition = "clean_ocr"
        degraded_condition = "degraded_ocr"
        clean = lookup[(engine_name, "verbatim", condition, "BM25 (Tuned)")]
        light = lookup[(engine_name, "light", condition, "BM25 (Tuned)")]
        degraded = lookup[(engine_name, "verbatim", degraded_condition, "BM25 (Tuned)")]
        drops.extend(
            [
                {
                    "engine": engine_name,
                    "comparison": "clean verbatim minus light paraphrase",
                    "clean_mrr": clean["mrr"],
                    "comparison_mrr": light["mrr"],
                    "drop_mrr": f"{float(clean['mrr']) - float(light['mrr']):.4f}",
                    "source": "results/v8/dose_response_reconciled.csv",
                },
                {
                    "engine": engine_name,
                    "comparison": "clean verbatim minus optical blur",
                    "clean_mrr": clean["mrr"],
                    "comparison_mrr": degraded["mrr"],
                    "drop_mrr": f"{float(clean['mrr']) - float(degraded['mrr']):.4f}",
                    "source": "results/v8/dose_response_reconciled.csv",
                },
            ]
        )

    write_csv(
        OUT / "dose_response_reconciled.csv",
        [
            "engine",
            "source_file",
            "source_row",
            "condition",
            "query_type",
            "system",
            "mrr",
            "ci_lower",
            "ci_upper",
        ],
        selected,
    )
    write_csv(
        OUT / "dose_response_drops_v8.csv",
        ["engine", "comparison", "clean_mrr", "comparison_mrr", "drop_mrr", "source"],
        drops,
    )


def generate_dense_reconciliation() -> None:
    rows = [
        {
            "quantity": "Dense MiniLM clean ground-truth verbatim",
            "value": "0.1919",
            "source_file": "results/v7/dose_response_table_v7.csv",
            "source_row": "4",
            "subset": "engine replication / short verbatim probe subset",
        },
        {
            "quantity": "Dense MiniLM clean ground-truth regime-grid verbatim",
            "value": "0.2100",
            "source_file": "results/v3/headline_regime_grid_v3.csv",
            "source_row": "4",
            "subset": "held-out regime grid / chunked MaxP",
        },
        {
            "quantity": "Dense MiniLM unchunked receipt text",
            "value": "0.2056",
            "source_file": "results/v3/dense_truncation_impact.csv",
            "source_row": "2",
            "subset": "SROIE receipt corpus / unchunked single-vector",
        },
        {
            "quantity": "Dense MiniLM chunked MaxP receipt text",
            "value": "0.2030",
            "source_file": "results/v3/dense_truncation_impact.csv",
            "source_row": "4",
            "subset": "SROIE receipt corpus / 150-word MaxP windows",
        },
    ]
    write_csv(OUT / "dense_reconciliation_v8.csv", ["quantity", "value", "source_file", "source_row", "subset"], rows)


def generate_analysis_status() -> None:
    source = RESULTS / "v7" / "analysis_status.csv"
    rows = read_csv(source)
    for row in rows:
        row["Evidence"] = row["Primary_Script_or_Source"]
        row["V8_Note"] = (
            "Classification follows the frozen 97d4ab56 boundary; post-freeze runs are confirmatory."
            if row["Status"] == "CONFIRMATORY"
            else "Classification follows the frozen 97d4ab56 boundary; pre-freeze run is exploratory."
        )
    write_csv(OUT / "analysis_status.csv", list(rows[0]), rows)


def generate_colqwen_status() -> None:
    rows = [
        {
            "item": "B1",
            "status": "BLOCKED",
            "evidence": "results/v7/colqwen_vs_proxies_paired_v7.csv",
            "reason": "Paired summary is against Word BM25 only; no per-query score vectors for grid-best hybrid/character proxies are present.",
        },
        {
            "item": "B2",
            "status": "BLOCKED",
            "evidence": "results/v7/colqwen_vs_proxies_paired_v7.csv; results/v3/headline_regime_grid_v3.csv",
            "reason": "Paired summary uses 0.9450 Word-BM25 proxy, while the grid best proxy is character BM25 0.9471; paired vectors for both baselines are absent.",
        },
        {
            "item": "B4",
            "status": "BLOCKED",
            "evidence": "results/v7/colqwen_receipts_measured_v7.csv",
            "reason": "Best-proxy tie sets require paired per-query scores that are not included in the checked-in summary artifacts.",
        },
    ]
    write_csv(OUT / "colqwen_comparison_status_v8.csv", ["item", "status", "evidence", "reason"], rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generate_dose_response()
    generate_dense_reconciliation()
    generate_analysis_status()
    generate_colqwen_status()
    shutil.copy2(RESULTS / "v7" / "paraphrase_llm_comparison_v7.csv", OUT / "paraphrase_llm_comparison_v8.csv")
    print(f"Generated V8 artifacts in {OUT}")


if __name__ == "__main__":
    main()
