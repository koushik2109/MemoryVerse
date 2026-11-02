#!/usr/bin/env python3
"""scripts/build_numeric_claims.py

Extracts all numeric claims from paper_v4/ijcds_body.tex across Abstract, Sections I-VIII,
Tables I-XV, and Figures 1-6.
Maps each claim to its source file in results/ or marks as UNVERIFIED.
Outputs numeric_claims.csv with columns:
value, location, metric, system, condition, split, pool, text_source, N, source_file, status
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
TEX_PATH = REPO_ROOT / "paper_v4" / "ijcds_body.tex"
HEADER_PATH = REPO_ROOT / "paper_v4" / "ijcds_header.tex"
OUT_CSV = REPO_ROOT / "numeric_claims.csv"
V4_OUT_DIR = REPO_ROOT / "results" / "v4" / "S-02"
PROV_PATH = REPO_ROOT / "results" / "v10" / "number_provenance.csv"

# Structural numbers to exclude (LaTeX syntax, standard parameters, years)
EXCLUDE_SET = {
    "2021", "2022", "2023", "2024", "2025", "2026",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "0.05", "0.10", "95%", "95", "90%", "90", "1.00", "0.00", "0.000",
    "42", "43", "44", "0.1", "0.2", "0.5", "1.0", "1.5", "2.0"
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
    """Search for the exact value string in results directory."""
    clean = val_str.replace("\\%", "").replace("%", "").replace(",", "").replace("{,}", "").strip()
    candidates = [clean]
    if clean.startswith("+") or clean.startswith("-"):
        candidates.append(clean[1:])
    
    # Search common csv/json in results
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
    lines = []
    if HEADER_PATH.exists():
        lines.extend(HEADER_PATH.read_text(encoding="utf-8").splitlines())
    lines.extend(TEX_PATH.read_text(encoding="utf-8").splitlines())
    prov_db = load_provenance()

    claims = []
    current_sec = "Preamble"
    in_table = False
    table_label = ""
    table_num = ""
    in_figure = False
    figure_label = ""
    figure_num = ""

    # Map labels to Table/Figure numbers
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
        "tab:pilot_summary": "Table XV",
    }

    fig_map = {
        "fig:architecture": "Figure 1",
        "fig:mrr_regime_systems": "Figure 2",
        "fig:dose_response_dual": "Figure 3",
        "fig:tokenization_fragmentation": "Figure 4a",
        "fig:embedding_drift": "Figure 4b",
        "fig:error_attribution": "Figure 4c",
        "fig:mechanism_plots": "Figure 4",
        "fig:conformal_pareto": "Figure 5",
        "fig:coco_native_vs_proxy": "Figure 6",
    }

    num_pattern = re.compile(r"(?<![a-zA-Z\\])[+-]?(?:\d{1,3}(?:(?:\{,\}|,)\d{3})+|\d+)(?:\.\d+)?%?(?:\\%)?")
    seen = set()

    for line_idx, line in enumerate(lines, 1):
        stripped = line.strip()

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

        if stripped.startswith("%"):
            continue

        # Clean non-claim macros
        cleaned_line = re.sub(r"\\url\{[^}]*\}", "", stripped)
        cleaned_line = re.sub(r"\\cite\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\ref\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\label\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\texttt\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", cleaned_line)
        cleaned_line = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", cleaned_line)

        for m in num_pattern.finditer(cleaned_line):
            raw_val = m.group(0).replace("{,}", "").replace(",", "").strip()
            clean_val = raw_val.lstrip("+-").rstrip("%").rstrip("\\%").strip()

            if clean_val in EXCLUDE_SET or raw_val in EXCLUDE_SET:
                continue

            # Deduplicate by value + line
            key = (raw_val, line_idx)
            if key in seen:
                continue
            seen.add(key)

            # Location label
            if in_table and table_num:
                loc = f"{table_num} (line {line_idx})"
            elif in_figure and figure_num:
                loc = f"{figure_num} (line {line_idx})"
            else:
                loc = f"{current_sec} (line {line_idx})"

            # Try to match provenance
            prov = prov_db.get(raw_val) or prov_db.get(clean_val) or prov_db.get(f"+{clean_val}") or prov_db.get(f"-{clean_val}")

            # Heuristics for system, metric, condition, N, pool, split, text_source
            context_window = cleaned_line[max(0, m.start()-40):min(len(cleaned_line), m.end()+40)]
            
            # Metric heuristic
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
                metric = "nDCG@10"
            elif "CER" in cleaned_line:
                metric = "CER"
            elif "WER" in cleaned_line:
                metric = "WER"
            elif "ms" in cleaned_line or "latency" in cleaned_line.lower():
                metric = "Latency (ms)"
            elif "p " in cleaned_line or "p=" in cleaned_line or "p<" in cleaned_line or "p_{\\text{Holm}}" in cleaned_line or "p_{\\text{TOST}}" in cleaned_line:
                metric = "p-value"
            elif "%" in raw_val or "coverage" in cleaned_line.lower() or "rate" in cleaned_line.lower():
                metric = "Percentage / Rate"
            elif clean_val in ["306", "813", "1626", "1000", "500", "50", "626", "2754", "600"]:
                metric = "Sample Count (N)"

            # System heuristic
            system = "General / Benchmark"
            if "ColQwen2" in cleaned_line or "ColQwen" in cleaned_line:
                system = "ColQwen2"
            elif "Char BM25" in cleaned_line or "Character" in cleaned_line or "Char" in cleaned_line:
                system = "Char BM25"
            elif "Word BM25" in cleaned_line or "Word" in cleaned_line:
                system = "Word BM25"
            elif "Hybrid" in cleaned_line:
                system = "Tuned Hybrid"
            elif "MiniLM" in cleaned_line:
                system = "Dense MiniLM"
            elif "BGE" in cleaned_line:
                system = "Dense BGE"
            elif "SigLIP" in cleaned_line:
                system = "SigLIP"
            elif "CLIP" in cleaned_line:
                system = "CLIP"
            elif "Florence" in cleaned_line:
                system = "Florence-2"
            elif "BLIP" in cleaned_line:
                system = "BLIP / BLIP-2"
            elif "PRF" in cleaned_line:
                system = "Extractive PRF"

            # Condition heuristic
            condition = "General"
            if "Clean Apple" in cleaned_line or "Apple Clean" in cleaned_line or "Apple Vision" in cleaned_line:
                condition = "Apple Clean OCR"
            elif "Degraded" in cleaned_line or "optical blur" in cleaned_line.lower() or "blur" in cleaned_line.lower():
                condition = "Optical Blur Degraded"
            elif "Tesseract" in cleaned_line:
                condition = "Tesseract 5.5 OCR"
            elif "EasyOCR" in cleaned_line:
                condition = "EasyOCR 1.7.2"
            elif "Ground Truth" in cleaned_line or "ground-truth" in cleaned_line.lower():
                condition = "Ground Truth Text"
            elif "light" in cleaned_line.lower() or "paraphrase" in cleaned_line.lower():
                condition = "Light Paraphrase"
            elif "answerable" in cleaned_line.lower():
                condition = "Answerable Question"
            elif "verbatim" in cleaned_line.lower():
                condition = "Verbatim"
            elif "20%" in cleaned_line:
                condition = "Synthetic 20% Noise"

            # Pool heuristic
            pool = "813 w/ COCO caption fallback"
            if "FUNSD" in cleaned_line or "Table X" in loc or "forms" in cleaned_line.lower():
                pool = "100 scanned forms"
            elif "DocVQA" in cleaned_line or "Table IX" in loc:
                pool = "500 document pages"
            elif "COCO" in cleaned_line or "Table VII" in loc or "captions" in cleaned_line.lower():
                pool = "1,000 Karpathy test images"
            elif "pure receipts" in cleaned_line.lower() or "626" in cleaned_line:
                pool = "626 pure receipts"
            elif "pilot" in loc.lower() or "Table XV" in loc:
                pool = "151 multimodal media"

            # Text source heuristic
            text_source = "Apple Clean"
            if "Ground Truth" in condition or "Clean GT" in cleaned_line or "Clean GT" in condition:
                text_source = "Clean GT"
            elif "Tesseract" in condition or "Tesseract" in cleaned_line:
                text_source = "Tesseract 5.5"
            elif "EasyOCR" in condition or "EasyOCR" in cleaned_line:
                text_source = "EasyOCR 1.7.2"
            elif "blur" in condition.lower() or "degraded" in condition.lower():
                text_source = "Apple Degraded Optical Blur"
            elif "Syn" in condition or "20%" in condition:
                text_source = "Syn-20%"
            elif system in ["ColQwen2", "CLIP", "SigLIP"]:
                text_source = "Pixel/Audio Native"

            # Split heuristic
            split = "Held-Out Test (50%)"
            if "Dev" in cleaned_line or "dev" in cleaned_line:
                split = "Dev (50%)"
            elif "full" in cleaned_line.lower() or "2754" in cleaned_line:
                split = "Full Split (100%)"

            # Sample size N
            N = "306"
            if "500" in cleaned_line or "DocVQA" in cleaned_line or pool == "500 document pages":
                N = "500"
            elif "1000" in cleaned_line or "1{,}000" in cleaned_line or "COCO" in cleaned_line or pool == "1,000 Karpathy test images":
                N = "1000"
            elif "100" in cleaned_line and ("human" in cleaned_line.lower() or "relational" in cleaned_line.lower() or pool == "100 scanned forms"):
                N = "100"
            elif "200" in cleaned_line and "human" in cleaned_line.lower():
                N = "200"
            elif "50" in cleaned_line and ("hop" in cleaned_line or "multi-hop" in cleaned_line):
                N = "50"
            elif "110" in cleaned_line or "audit" in cleaned_line.lower():
                N = "110"
            elif "626" in cleaned_line:
                N = "626"
            elif "2754" in cleaned_line or "2{,}754" in cleaned_line:
                N = "2754"
            elif "813" in cleaned_line:
                N = "813"
            elif "1626" in cleaned_line or "1{,}626" in cleaned_line:
                N = "1626"

            source_file = ""
            status = "UNVERIFIED"

            if prov:
                source_file = prov.get("source_file", "")
                if prov.get("metric_description"):
                    metric = prov.get("metric_description")
                if prov.get("query_type"):
                    condition = f"{prov.get('query_type')} / {prov.get('text_source', condition)}"
                if prov.get("text_source"):
                    text_source = prov.get("text_source")
                status = "VERIFIED"
            else:
                # Fallback search in results dir
                res_match = find_in_results_dir(raw_val)
                if res_match:
                    source_file = res_match
                    status = "VERIFIED"

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

    V4_OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["value", "location", "metric", "system", "condition", "split", "pool", "text_source", "N", "source_file", "status"]
    
    for target_path in [OUT_CSV, V4_OUT_DIR / "numeric_claims.csv"]:
        with open(target_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in claims:
                writer.writerow(c)

    print(f"Total claims extracted: {len(claims)}")
    verified = sum(1 for c in claims if c["status"] == "VERIFIED")
    unverified = sum(1 for c in claims if c["status"] == "UNVERIFIED")
    print(f"VERIFIED: {verified}, UNVERIFIED: {unverified}")

    # Write untraceable claims report
    unverified_claims = [c for c in claims if c["status"] == "UNVERIFIED"]
    untraceable_md = V4_OUT_DIR / "untraceable_claims_report.md"
    with open(untraceable_md, mode="w", encoding="utf-8") as f:
        f.write("# Untraceable Claims Report (S-02)\n\n")
        f.write(f"- Total Claims: {len(claims)}\n")
        f.write(f"- Verified: {verified}\n")
        f.write(f"- Untraceable / Unverified: {len(unverified_claims)}\n\n")
        f.write("| Value | Location | Metric | System | Condition | Pool | Text Source | N |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for u in unverified_claims:
            f.write(f"| `{u['value']}` | {u['location']} | {u['metric']} | {u['system']} | {u['condition']} | {u['pool']} | {u['text_source']} | {u['N']} |\n")

    print(f"Saved untraceable report to: {untraceable_md}")

if __name__ == "__main__":
    main()
