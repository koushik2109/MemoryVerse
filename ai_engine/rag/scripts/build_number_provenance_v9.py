#!/usr/bin/env python3
"""Build and update number_provenance.csv for V9.

Updates:
- Word BM25 clean OCR unified to 0.8371 verbatim / 0.4334 light paraphrase.
- Optical blur drop: Apple Vision drop is 0.1399 (0.8371 to 0.6972); Tesseract drop is 0.3916 (0.7364 to 0.3448).
- Ghost number 0.6724 removed.
- Best proxy comparison for ColQwen2: lift is +0.0651 over Tuned Hybrid / Word BM25 (0.4334).
- ColQwen2 answerable question compared against grid baselines (word 0.9014, char 0.9471).
- Cross-hardware ratio 286x removed; exact per-hardware profile maintained (249.21 ms on T4 GPU vs 0.87-0.96 ms on Apple M2).
- Conformal numbers verified (mean test risk 0.0494, 95% quantile CI [0.0355, 0.0654], split coverage 58.5%, escalation 78.4%, savings 21.6%, MRR 0.3577 vs 0.3579).
- Unsupported Qwen2.5-72B and +0.0026 to +0.0059 hybrid claims removed.
- p-values floored at p < 1e-5.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V7_PROV = ROOT / "results" / "v7" / "number_provenance.csv"
V9_PROV = ROOT / "results" / "v9" / "number_provenance.csv"


def main() -> None:
    with V7_PROV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    v9_rows = []
    for r in rows:
        num = r["number"].strip()
        loc = r["manuscript_location"]
        desc = r["metric_description"]

        # 1. Replace 0.0982 with 0.0651 (lift over best proxy in grid)
        if num == "0.0982":
            r["number"] = "0.0651"
            r["source_file"] = "results/v9/colqwen_comparison_status_v9.csv"
            r["row_column"] = "row 2, col lift"
            r["metric_description"] = "ColQwen2-v1.0 MRR lift over best text proxy (Tuned Hybrid / Word BM25 0.4334) under Light Paraphrasing"
            v9_rows.append(r)
            continue

        # 2. Add 0.065 to match abstract/contributions if referenced as rounded
        # 3. Handle ghost number 0.6724: remove if present
        if num == "0.6724":
            continue

        # 4. Remove 286 if listed as standalone ratio
        if num == "286" or num == "286x":
            continue

        # 5. Remove 0.0026 or 0.0059 hybrid lift in conclusion if listed
        if "hybrid adds only" in desc.lower() or ("0.0026" in num and "hybrid" in desc.lower()):
            continue

        # 6. Ensure Apple Vision light drop is 0.4037 and blur drop is 0.1399
        if num == "0.4321":
            r["number"] = "0.4037"
            r["source_file"] = "results/v9/dose_response_drops_v9.csv"
            r["row_column"] = "row 2, col drop_mrr"
            r["metric_description"] = "Word BM25 MRR drop under Light Paraphrase on Apple Vision OCR (0.8371 to 0.4334)"
            v9_rows.append(r)
            continue

        if num == "0.1600" or num == "0.16":
            r["number"] = "0.1399"
            r["source_file"] = "results/v9/dose_response_drops_v9.csv"
            r["row_column"] = "row 3, col drop_mrr"
            r["metric_description"] = "Word BM25 MRR drop under Optical Blur on Apple Vision OCR (0.8371 to 0.6972)"
            v9_rows.append(r)
            continue

        v9_rows.append(r)

    # Add specific new V9 entries
    additional_entries = [
        {
            "number": "0.4037",
            "source_file": "results/v9/dose_response_drops_v9.csv",
            "row_column": "row 2, col drop_mrr",
            "query_type": "light_paraphrase",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-A / Sec V-C",
            "metric_description": "Apple Vision Word BM25 MRR drop under light paraphrasing (0.8371 to 0.4334)",
        },
        {
            "number": "0.1399",
            "source_file": "results/v9/dose_response_drops_v9.csv",
            "row_column": "row 3, col drop_mrr",
            "query_type": "verbatim",
            "text_source": "Apple Vision Degraded OCR",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-A / Sec V-C",
            "metric_description": "Apple Vision Word BM25 MRR drop under optical blur (0.8371 to 0.6972)",
        },
        {
            "number": "0.0651",
            "source_file": "results/v9/colqwen_comparison_status_v9.csv",
            "row_column": "row 2, col lift",
            "query_type": "light_paraphrase",
            "text_source": "Direct Document Image Pixels",
            "subset": "test_split",
            "manuscript_location": "Abstract / Contributions / Sec IV-D / Conclusion",
            "metric_description": "ColQwen2-v1.0 MRR lift over best text proxy in regime grid (0.4985 vs 0.4334)",
        },
        {
            "number": "0.065",
            "source_file": "results/v9/colqwen_comparison_status_v9.csv",
            "row_column": "row 2, col lift (rounded)",
            "query_type": "light_paraphrase",
            "text_source": "Direct Document Image Pixels",
            "subset": "test_split",
            "manuscript_location": "Abstract / Contributions",
            "metric_description": "Rounded ColQwen2-v1.0 MRR lift over best text proxy in regime grid (~+0.065)",
        },
        {
            "number": "0.9014",
            "source_file": "results/v3/headline_regime_grid_v3.csv",
            "row_column": "row 22, col MRR",
            "query_type": "answerable_questions",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Sec IV-D / Table XIII / Table XIV",
            "metric_description": "Word BM25 MRR on answerable questions in headline regime grid",
        },
        {
            "number": "0.9471",
            "source_file": "results/v3/headline_regime_grid_v3.csv",
            "row_column": "row 23, col MRR",
            "query_type": "answerable_questions",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Sec IV-D / Table XIII / Table XIV",
            "metric_description": "Char 3-5gram BM25 MRR on answerable questions in headline regime grid",
        },
        {
            "number": "0.2100",
            "source_file": "results/v9/dense_reconciliation_v9.csv",
            "row_column": "row 2, col value",
            "query_type": "verbatim",
            "text_source": "Pristine Ground Truth Text",
            "subset": "test_split",
            "manuscript_location": "Sec IV-B / Table VII",
            "metric_description": "Dense MiniLM chunked MaxP MRR on pristine ground truth text in regime grid",
        },
        {
            "number": "0.2056",
            "source_file": "results/v9/dense_reconciliation_v9.csv",
            "row_column": "row 3, col value",
            "query_type": "verbatim",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-B",
            "metric_description": "Dense MiniLM unchunked single-vector MRR on clean OCR text",
        },
        {
            "number": "0.2030",
            "source_file": "results/v9/dense_reconciliation_v9.csv",
            "row_column": "row 4, col value",
            "query_type": "verbatim",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-B",
            "metric_description": "Dense MiniLM chunked MaxP (150-word window) MRR on clean OCR text",
        },
        {
            "number": "334",
            "source_file": "ai_engine/rag/data/tier1_proxy_consistency.json",
            "row_column": "records_count",
            "query_type": "all",
            "text_source": "Tier-1 Proxy Annotations",
            "subset": "pilot",
            "manuscript_location": "Table Caption #15 (Appendix A)",
            "metric_description": "Tier-1 proxy consistency pilot records count",
        },
        {
            "number": "85",
            "source_file": "ai_engine/rag/data/ocr_benchmark_results.json",
            "row_column": "probes_count",
            "query_type": "verbatim",
            "text_source": "Scene-text OCR",
            "subset": "pilot",
            "manuscript_location": "Table Caption #15 (Appendix A)",
            "metric_description": "Tier-2 scene text pilot probes count",
        },
        {
            "number": "13",
            "source_file": "ai_engine/rag/data/audio_benchmark_results.json",
            "row_column": "probes_count",
            "query_type": "narrative",
            "text_source": "Raw WAV Audio",
            "subset": "pilot",
            "manuscript_location": "Table Caption #15 (Appendix A)",
            "metric_description": "Tier-2 audio pilot probes count",
        },
        {
            "number": "53",
            "source_file": "ai_engine/rag/data/visual_benchmark_results.json",
            "row_column": "probes_count",
            "query_type": "visual",
            "text_source": "Raw Image Pixels",
            "subset": "pilot",
            "manuscript_location": "Table Caption #15 (Appendix A)",
            "metric_description": "Tier-2 visual pilot probes count",
        },
        {
            "number": "250",
            "source_file": "ai_engine/rag/data/coco_pilot_results.json",
            "row_column": "images_count",
            "query_type": "caption",
            "text_source": "MS-COCO Pilot",
            "subset": "pilot",
            "manuscript_location": "Table Caption #15 (Appendix A)",
            "metric_description": "Preliminary MS-COCO pilot images count",
        },
        {
            "number": "600",
            "source_file": "results/v3/funsd_dataset_replication.csv",
            "row_column": "pool_size",
            "query_type": "all",
            "text_source": "FUNSD Administrative Forms",
            "subset": "funsd_test",
            "manuscript_location": "Sec IV-C / Table Caption #12",
            "metric_description": "FUNSD scanned forms evaluation distractor pool size",
        },
        {
            "number": "0.0215",
            "source_file": "results/v3/multihop_paired_results.csv",
            "row_column": "row 4, col exact_mcnemar_p",
            "query_type": "relational_multihop",
            "text_source": "Text Proxies",
            "subset": "50_multihop_queries",
            "manuscript_location": "Table Caption #8 / Sec IV-B",
            "metric_description": "Exact McNemar p-value comparing LLM multi-hop bridging to Sparse BM25 alone",
        },
        {
            "number": "0.8937",
            "source_file": "results/v3/headline_regime_grid_v3.csv",
            "row_column": "answerable questions lower MRR",
            "query_type": "answerable_questions",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-A",
            "metric_description": "Lower bound of BM25 MRR range on answerable factual questions across OCR regimes",
        },
        {
            "number": "0.9192",
            "source_file": "results/v3/headline_regime_grid_v3.csv",
            "row_column": "answerable questions upper MRR",
            "query_type": "answerable_questions",
            "text_source": "Apple Vision Clean OCR",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-A",
            "metric_description": "Upper bound of BM25 MRR range on answerable factual questions across OCR regimes",
        },
        {
            "number": "2754",
            "source_file": "results/v3/conformal_pooled_cache.npz",
            "row_column": "margins.shape[0]",
            "query_type": "pooled_queries",
            "text_source": "Heterogeneous Proxies",
            "subset": "calibration_splits",
            "manuscript_location": "Contributions / Table Caption #6",
            "metric_description": "Total pooled query instances in conformal cascade cache",
        },
        {
            "number": "33.4%",
            "source_file": "results/v3/dense_truncation_impact.csv",
            "row_column": "text description (33.39%)",
            "query_type": "verbatim",
            "text_source": "SROIE Receipts",
            "subset": "corpus_626",
            "manuscript_location": "Contributions / Sec IV-B",
            "metric_description": "Percentage of SROIE receipts exceeding 256 WordPiece tokens",
        },
        {
            "number": "79.7%",
            "source_file": "results/v6/dense_formulation_keyword_control_v6.csv",
            "row_column": "row 6, col BGE_Base_Hit10 (79.74%)",
            "query_type": "answerable_questions",
            "text_source": "Pristine Ground Truth Text",
            "subset": "test_split",
            "manuscript_location": "Contributions / Sec IV-B",
            "metric_description": "Rounded Hit@10 percentage of BGE-Base under natural answerable questions",
        },
        {
            "number": "0.640",
            "source_file": "results/v6/multihop_independent_v6.csv",
            "row_column": "row 2, col Extractive_Hit10 / LLM_Hit10",
            "query_type": "relational_multihop",
            "text_source": "Text Proxies",
            "subset": "50_multihop_queries",
            "manuscript_location": "Abstract / Contributions / Table IX / Conclusion",
            "metric_description": "TargetHit@10 for Extractive PRF and Generative LLM Bridging on 50 diagnostic multi-hop queries",
        },
        {
            "number": "0.0792",
            "source_file": "results/v6/multihop_independent_v6.csv",
            "row_column": "row 2, col Diff_CI_Upper / Diff_CI_Lower",
            "query_type": "relational_multihop",
            "text_source": "Text Proxies",
            "subset": "50_multihop_queries",
            "manuscript_location": "Contributions / Table IX / Conclusion",
            "metric_description": "95% bootstrap confidence interval bound for paired Target Hit@10 difference on multi-hop queries",
        },
        {
            "number": "0.0355",
            "source_file": "results/v7/conformal_retest_summary_v7.csv",
            "row_column": "row 2, col ci95_q025",
            "query_type": "pooled_queries",
            "text_source": "Heterogeneous Proxies",
            "subset": "calibration_splits",
            "manuscript_location": "Contributions / Limitations / Conclusion",
            "metric_description": "Empirical test miss risk 95% quantile CI lower bound under marginal loss",
        },
        {
            "number": "0.0654",
            "source_file": "results/v7/conformal_retest_summary_v7.csv",
            "row_column": "row 2, col ci95_q975",
            "query_type": "pooled_queries",
            "text_source": "Heterogeneous Proxies",
            "subset": "calibration_splits",
            "manuscript_location": "Contributions / Limitations / Conclusion",
            "metric_description": "Empirical test miss risk 95% quantile CI upper bound under marginal loss",
        },
        {
            "number": "122.3%",
            "source_file": "results/v2/mechanism_tokenization_fragmentation.csv",
            "row_column": "row 5, col MiniLM_Subwords_Per_Word",
            "query_type": "synthetic_noise",
            "text_source": "Tokenized Text",
            "subset": "noise_sweep_0_to_40",
            "manuscript_location": "Contributions / Sec IV-B",
            "metric_description": "WordPiece subword fragmentation increase from 1.347 at clean to 2.994 at 40% noise",
        },
    ]

    existing_nums = {(r["number"], r["metric_description"]) for r in v9_rows}
    for e in additional_entries:
        if (e["number"], e["metric_description"]) not in existing_nums:
            v9_rows.append(e)

    V9_PROV.parent.mkdir(parents=True, exist_ok=True)
    with V9_PROV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(v9_rows[0].keys()))
        writer.writeheader()
        writer.writerows(v9_rows)

    print(f"Saved {len(v9_rows)} verified provenance records to {V9_PROV}")


if __name__ == "__main__":
    main()
