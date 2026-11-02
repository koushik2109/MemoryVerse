#!/usr/bin/env python3
"""
scripts/update_table_viii_colqwen.py — Automatic Table VIII Updater from Measured Kaggle Results
================================================================================================
Reads results/v13_colqwen_robustness/colqwen_robustness_summary.csv, updates Table VIII in
paper_v3/ijcds_body.tex (and copies), compiles with Tectonic, and verifies strict 15-page limit.
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "v13_colqwen_robustness" / "colqwen_robustness_summary.csv"
TEX_PATH = ROOT / "paper_v3" / "ijcds_body.tex"


def update_table(summary_path: Path):
    if not summary_path.exists():
        print(f"Error: Summary CSV not found at {summary_path}")
        print("Please place the downloaded colqwen_robustness_summary.csv into results/v13_colqwen_robustness/")
        sys.exit(1)

    df = pd.read_csv(summary_path)
    print("Loaded Summary CSV:")
    print(df)

    # Extract ColQwen values
    val_map = {}
    for _, row in df.iterrows():
        src = str(row["Query_Source"])
        mrr = float(row["ColQwen_MRR"])
        if "Primary" in src:
            val_map["primary"] = mrr
        elif "Qwen" in src:
            val_map["qwen"] = mrr
        elif "100" in src:
            val_map["human_100"] = mrr
        elif "200" in src:
            val_map["human_200"] = mrr

    print("\nExtracted ColQwen MRR values:")
    for k, v in val_map.items():
        print(f"  {k}: {v:.4f}")

    if not all(k in val_map for k in ["qwen", "human_100", "human_200"]):
        print("Error: Missing one of the required query suites in summary CSV.")
        sys.exit(1)

    tex_text = TEX_PATH.read_text(encoding="utf-8")

    # Replacement targets in Table VIII
    target_qwen = "Secondary Gen.\\ (\\texttt{Qwen2.5-7B}, $N=306$) & 0.6292 & 0.5802 & 0.2685 & ---\\rlap{$^*$} \\\\"
    replace_qwen = f"Secondary Gen.\\ (\\texttt{{Qwen2.5-7B}}, $N=306$) & 0.6292 & 0.5802 & 0.2685 & {val_map['qwen']:.4f} \\\\"

    target_h100 = "Human-Authored Queries ($N=100$)             & 0.6321 & 0.5758 & 0.5020 & ---\\rlap{$^*$} \\\\"
    replace_h100 = f"Human-Authored Queries ($N=100$)             & 0.6321 & 0.5758 & 0.5020 & {val_map['human_100']:.4f} \\\\"

    target_h200 = "Human-Authored Queries ($N=200$)             & 0.5766 & 0.5549 & 0.4909 & ---\\rlap{$^*$} \\\\"
    replace_h200 = f"Human-Authored Queries ($N=200$)             & 0.5766 & 0.5549 & 0.4909 & {val_map['human_200']:.4f} \\\\"

    target_fn = "\\raggedright\\scriptsize $^*$ColQwen2-v1.0 was profiled exclusively on the primary frozen synthetic test split ($N=306$ probes, MRR = 0.4962 on Tesla T4) and was not re-evaluated across secondary or human query sets."
    replace_fn = "\\raggedright\\scriptsize Note: All systems evaluated across identical held-out document image pools ($N=813$). ColQwen2-v1.0 evaluated on Nvidia Tesla T4 GPU."

    if target_qwen not in tex_text:
        print("Warning: Target Table VIII lines not found in ijcds_body.tex. Already updated?")
    else:
        tex_text = tex_text.replace(target_qwen, replace_qwen)
        tex_text = tex_text.replace(target_h100, replace_h100)
        tex_text = tex_text.replace(target_h200, replace_h200)
        tex_text = tex_text.replace(target_fn, replace_fn)

        TEX_PATH.write_text(tex_text, encoding="utf-8")
        (ROOT / "ijcds" / "ijcds_body.tex").write_text(tex_text, encoding="utf-8")
        (ROOT / "ijcds_body.tex").write_text(tex_text, encoding="utf-8")
        print("\nSuccessfully updated Table VIII in LaTeX sources!")

    # Compile with Tectonic
    print("\nCompiling paper_v3/main.pdf with Tectonic...")
    res = subprocess.run(["tectonic", "main.tex"], cwd=str(ROOT / "paper_v3"), capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation failed:")
        print(res.stderr[:1000])
        sys.exit(1)

    # Verify Page Count via osascript
    jxa = """
    ObjC.import("PDFKit");
    var pdf = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath("/Users/anuragkrishnavennela/Projects/MemoryVerse/paper_v3/main.pdf"));
    console.log("Total pages: " + pdf.pageCount);
    """
    res_jxa = subprocess.run(["osascript", "-l", "JavaScript", "-e", jxa], capture_output=True, text=True)
    print(res_jxa.stdout.strip())
    if "Total pages: 15" not in res_jxa.stdout:
        print("WARNING: Page count is NOT 15!")
    else:
        print("VERIFIED: Exactly 15 pages!")


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else SUMMARY_CSV
    update_table(p)
