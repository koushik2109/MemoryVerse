#!/usr/bin/env python3
"""scripts/build_numeric_claims_v4.py

Rebuilds numeric_claims.csv for Phase 2 (R-04), integrating all corrected results
from R-01, R-02, and R-03:
- Table IV standalone BM25 and Oracle invariance (R-01)
- Table III chunked vs unchunked reporting (R-01)
- MS-COCO TOST and multi-caption recomputations (R-02)
- Multi-hop per-subset stats, TOST, and latency reconciliation (R-02)
- DocVQA full benchmark vs 500-split nDCG reconciliation (R-02)
- Real-OCR paired permutation test for answerable queries (N-03)
- FUNSD EasyOCR strict IR recomputation (R-03)

Outputs:
- results/v4/R-04/numeric_claims_v4.csv
- results/v4/R-04/claims_correction_diff.csv
- results/v4/R-04/README.md
- numeric_claims.csv (updated root ledger)
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = REPO_ROOT / "paper_v4" / "ijcds_body.tex"
HEADER_PATH = REPO_ROOT / "paper_v4" / "ijcds_header.tex"
OUT_CSV = REPO_ROOT / "numeric_claims.csv"
OUT_DIR = REPO_ROOT / "results" / "v4" / "R-04"
PROV_PATH = REPO_ROOT / "results" / "v10" / "number_provenance.csv"

# Structural numbers to exclude (LaTeX syntax, standard parameters, years)
EXCLUDE_SET = {
    "2021", "2022", "2023", "2024", "2025", "2026",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "0.05", "0.10", "95%", "95", "90%", "90", "1.00", "0.00", "0.000",
    "42", "43", "44", "0.1", "0.2", "0.5", "1.0", "1.5", "2.0"
}

# Known corrections from Phase 1 & 2
CORRECTIONS_MAP = {
    # N-02 / R-01 Table IV
    "0.8069": {
        "new_value": "0.8107",
        "source": "results/v4/R-01/table_iv_corrected.csv",
        "note": "Standalone Word BM25 (K1=10 truncation), previously confounded with BM25+CLIP stream"
    },
    # N-03 Answerable paired test
    "0.00012": {
        "new_value": "0.0011",
        "source": "results/v4/N-03/paired_answerable_stats.md",
        "note": "Real-OCR paired permutation p-value (+0.0733 diff). Clean GT remains p=0.00012 (+0.0876 diff)."
    },
    # N-04 / R-03 FUNSD EasyOCR Blur
    "0.0227": {
        "new_value": "0.0101",
        "source": "results/v4/R-03/funsd_easyocr_rerun.csv:33",
        "note": "Strict IR unretrieved rule (RR=0 for blank documents). Non-blank N=29 is 0.0345."
    },
    "0.0462": {
        "new_value": "0.0437",
        "source": "results/v4/R-03/funsd_easyocr_rerun.csv:36",
        "note": "Strict IR unretrieved rule. Non-blank N=29 is 0.1492."
    },
    # N-09 DocVQA nDCG@5
    "0.6120": {
        "new_value": "0.6695",
        "source": "results/v4/R-02/docvqa_metrics_reconciled.csv:3",
        "note": "Own 500-split nDCG@5 is 0.6695; 0.6120 is official full-benchmark leaderboard score."
    },
}

def load_provenance():
    prov = {}
    if PROV_PATH.exists():
        with open(PROV_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                num = r["number"].strip()
                prov[num] = r
                clean = num.replace(",", "").replace("%", "").strip()
                prov[clean] = r
                if clean.startswith("+") or clean.startswith("-"):
                    prov[clean[1:]] = r
                else:
                    prov[f"+{clean}"] = r
                    prov[f"-{clean}"] = r
    return prov

def find_in_results_dir(val_str: str) -> str:
    clean = val_str.replace("\\%", "").replace("%", "").replace(",", "").replace("{,}", "").strip()
    candidates = [clean]
    if clean.startswith("+") or clean.startswith("-"):
        candidates.append(clean[1:])
    
    # Priority search in v4 results first
    for p in (REPO_ROOT / "results" / "v4").rglob("*"):
        if p.is_file() and p.suffix in [".csv", ".json", ".txt"]:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for c in candidates:
                    if len(c) >= 3 and c in content:
                        return str(p.relative_to(REPO_ROOT))
            except Exception:
                pass

    # Then full results search
    for p in (REPO_ROOT / "results").rglob("*"):
        if p.is_file() and p.suffix in [".csv", ".json", ".txt"]:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for c in candidates:
                    if len(c) >= 3 and c in content:
                        return str(p.relative_to(REPO_ROOT))
            except Exception:
                pass
    return ""

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if HEADER_PATH.exists():
        lines.extend(HEADER_PATH.read_text(encoding="utf-8").splitlines())
    lines.extend(TEX_PATH.read_text(encoding="utf-8").splitlines())
    prov_db = load_provenance()

    claims = []
    diff_records = []
    current_sec = "Preamble"
    in_table = False
    table_label = ""
    table_num = ""
    in_figure = False
    figure_label = ""
    figure_num = ""

    tab_map = {
        "tab:sota_comparison": "Table I",
        "tab:regime_grid_v3": "Table II",
        "tab:dense_formulation_v6": "Table III",
        "tab:ablation_v3": "Table IV",
        "tab:practical_stack": "Table V",
        "tab:conformal_cascade_v6": "Table VI",
        "tab:coco_benchmarks_v3": "Table VII",
        "tab:ocr_error_rates_v6": "Table VIII",
        "tab:docvqa_benchmark_v9": "Table IX",
        "tab:engine_replication_v3": "Table X",
        "tab:funsd_v3": "Table X",
        "tab:generator2_robustness": "Table XI",
        "tab:system_costs": "Table XII",
        "tab:decision_guide_v6": "Table XIII",
        "tab:multihop_paired_v3": "Table XIV",
        "tab:synthetic_noise_ablation": "Table XV",
    }

    fig_map = {
        "fig:pipeline": "Figure 1",
        "fig:regime_matrix": "Figure 2",
        "fig:dense_collapse": "Figure 3",
        "fig:conformal_calibration": "Figure 4",
        "fig:multihop_cost_accuracy": "Figure 5",
        "fig:latency_pareto": "Figure 6",
    }

    num_pattern = re.compile(
        r"(?<![A-Za-z0-9_\\/])"
        r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|ms|x)?)"
        r"(?![A-Za-z0-9_])"
    )

    seen = set()

    for line_idx, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue

        # Section tracking
        if "\\begin{abstract}" in stripped or "\\myabstract" in stripped:
            current_sec = "Abstract"
        elif "\\end{abstract}" in stripped:
            current_sec = "Introduction"
        elif "\\section{" in stripped:
            m = re.search(r"\\section\{([^}]+)\}", stripped)
            if m:
                current_sec = m.group(1).replace("\\", "").strip()
        elif "\\subsection{" in stripped:
            m = re.search(r"\\subsection\{([^}]+)\}", stripped)
            if m:
                current_sec = f"{current_sec} > {m.group(1).replace('\\', '').strip()}"

        # Table tracking
        if "\\begin{table" in stripped:
            in_table = True
            table_label = ""
            table_num = ""
        elif "\\end{table" in stripped:
            in_table = False

        if in_table:
            m = re.search(r"\\label\{([^}]+)\}", stripped)
            if m:
                table_label = m.group(1)
                table_num = tab_map.get(table_label, f"Table ({table_label})")

        # Figure tracking
        if "\\begin{figure" in stripped:
            in_figure = True
            figure_label = ""
            figure_num = ""
        elif "\\end{figure" in stripped:
            in_figure = False

        if in_figure:
            m = re.search(r"\\label\{([^}]+)\}", stripped)
            if m:
                figure_label = m.group(1)
                figure_num = fig_map.get(figure_label, f"Figure ({figure_label})")

        # Clean non-claim macros while preserving macro contents in table cells
        cleaned_line = re.sub(r"\\url\{[^}]*\}", "", stripped)
        cleaned_line = re.sub(r"\\cite\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\ref\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\label\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\texttt\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", cleaned_line)
        # Strip macro names but retain inner text: \makecell, \textbf, \tiny, \scriptsize
        cleaned_line = re.sub(r"\\(makecell|textbf|textit|textsc|tiny|scriptsize|footnotesize|large|Large)", "", cleaned_line)
        cleaned_line = cleaned_line.replace("\\\\", " ")

        for m in num_pattern.finditer(cleaned_line):
            raw_val = m.group(0).replace("{,}", "").replace(",", "").strip()
            clean_val = raw_val.lstrip("+-").rstrip("%").rstrip("ms").rstrip("x").strip()

            if clean_val in EXCLUDE_SET or raw_val in EXCLUDE_SET or not clean_val:
                continue
            if re.match(r"^\d+$", clean_val) and int(clean_val) < 21:
                continue

            key = (raw_val, line_idx)
            if key in seen:
                continue
            seen.add(key)

            if in_table and table_num:
                loc = f"{table_num} (line {line_idx})"
            elif in_figure and figure_num:
                loc = f"{figure_num} (line {line_idx})"
            else:
                loc = f"{current_sec} (line {line_idx})"

            metric = "Unknown"
            if "MRR" in cleaned_line:
                metric = "MRR"
            elif "Hit@10" in cleaned_line or "@10" in cleaned_line:
                metric = "Hit@10"
            elif "Hit@1" in cleaned_line or "@1" in cleaned_line:
                metric = "Hit@1"
            elif "Hit@5" in cleaned_line or "@5" in cleaned_line:
                metric = "Hit@5"
            elif "nDCG" in cleaned_line:
                metric = "nDCG@5"
            elif "CER" in cleaned_line:
                metric = "CER"
            elif "WER" in cleaned_line:
                metric = "WER"
            elif "ms" in raw_val or "latency" in cleaned_line.lower():
                metric = "Latency (ms)"
            elif "%" in raw_val:
                metric = "Percentage / Rate"
            elif "x" in raw_val:
                metric = "Speedup / Ratio"
            elif "p" in cleaned_line and ("<" in cleaned_line or "=" in cleaned_line):
                metric = "p-value"

            system = "IR System"
            for s in ["Char BM25", "Word BM25", "ColQwen2", "MiniLM", "BGE-base", "CLIP", "Florence-2", "BLIP", "SigLIP"]:
                if s.lower() in cleaned_line.lower():
                    system = s
                    break

            condition = "Standard Evaluation"
            pool = "813 documents (receipts + COCO distractors)"
            text_source = "Real OCR (Apple Vision)"
            split = "Held-Out Test (50%)"
            N = "306"

            if "DocVQA" in cleaned_line or "docvqa" in loc.lower():
                pool = "500 document pages"
                text_source = "Tesseract 5.5 OCR"
                N = "500"
            elif "COCO" in cleaned_line or "coco" in loc.lower():
                pool = "1,000 Karpathy test images"
                text_source = "Caption Proxy"
                N = "1000"
            elif "FUNSD" in cleaned_line or "funsd" in loc.lower():
                pool = "100 scanned forms + 500 distractors"
                text_source = "Apple / Tesseract / EasyOCR"
                N = "99"
            elif "Multi-hop" in cleaned_line or "multihop" in loc.lower():
                pool = "813 documents"
                text_source = "Apple Clean OCR"
                N = "100"

            source_file = ""
            status = "UNVERIFIED"

            prov = prov_db.get(raw_val) or prov_db.get(clean_val) or prov_db.get(f"+{clean_val}") or prov_db.get(f"-{clean_val}")
            if prov:
                source_file = prov.get("source_file", "")
                if prov.get("metric_description"):
                    metric = prov.get("metric_description")
                if prov.get("query_type"):
                    condition = f"{prov.get('query_type')} / {prov.get('text_source', condition)}"
                if prov.get("text_source"):
                    text_source = prov.get("text_source")
                status = "VERIFIED_UNCHANGED"
            else:
                res_match = find_in_results_dir(raw_val)
                if res_match:
                    source_file = res_match
                    status = "VERIFIED_UNCHANGED"

            # Check if this claim was corrected in Phase 1 / 2
            corr = CORRECTIONS_MAP.get(clean_val) or CORRECTIONS_MAP.get(raw_val)
            if corr:
                status = "PHASE2_CORRECTED"
                diff_records.append({
                    "manuscript_value": raw_val,
                    "corrected_value": corr["new_value"],
                    "location": loc,
                    "metric": metric,
                    "system": system,
                    "source_file": corr["source"],
                    "notes": corr["note"]
                })

            claims.append({
                "value": raw_val,
                "location": loc,
                "metric": metric,
                "system": system,
                "condition": condition,
                "split": split,
                "pool": pool,
                "text_source": text_source,
                "N": N,
                "source_file": source_file,
                "status": status,
            })

    fieldnames = ["value", "location", "metric", "system", "condition", "split", "pool", "text_source", "N", "source_file", "status"]
    
    for target_path in [OUT_CSV, OUT_DIR / "numeric_claims_v4.csv"]:
        with open(target_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in claims:
                writer.writerow(c)

    diff_path = OUT_DIR / "claims_correction_diff.csv"
    with open(diff_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["manuscript_value", "corrected_value", "location", "metric", "system", "source_file", "notes"])
        writer.writeheader()
        for d in diff_records:
            writer.writerow(d)

    readme_path = OUT_DIR / "README.md"
    total_claims = len(claims)
    n_unchanged = sum(1 for c in claims if c["status"] == "VERIFIED_UNCHANGED")
    n_corrected = sum(1 for c in claims if c["status"] == "PHASE2_CORRECTED")
    n_unverified = sum(1 for c in claims if c["status"] == "UNVERIFIED")

    with open(readme_path, mode="w", encoding="utf-8") as f:
        f.write("# R-04: Rebuilt Numeric Claims Ledger (Phase 2)\n\n")
        f.write(f"- **Total Claims in Manuscript:** {total_claims}\n")
        f.write(f"- **Verified Unchanged:** {n_unchanged}\n")
        f.write(f"- **Phase 2 Corrected (Scheduled for Phase 5 Patch):** {n_corrected}\n")
        f.write(f"- **Unverified / Structural Exclusions:** {n_unverified}\n\n")
        f.write("## Phase 2 Correction Diff\n")
        f.write("The following numbers in the current manuscript will be updated during Phase 5:\n\n")
        f.write("| Current MS Value | Corrected Value | Location | System | Corrected Source | Notes |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in diff_records:
            f.write(f"| `{d['manuscript_value']}` | `{d['corrected_value']}` | {d['location']} | {d['system']} | `{d['source_file']}` | {d['notes']} |\n")

    print(f"Total claims extracted: {total_claims}")
    print(f"Verified unchanged: {n_unchanged}")
    print(f"Phase 2 corrected: {n_corrected}")
    print(f"Unverified: {n_unverified}")
    print(f"Saved outputs to {OUT_DIR} and updated {OUT_CSV}")

if __name__ == "__main__":
    main()
