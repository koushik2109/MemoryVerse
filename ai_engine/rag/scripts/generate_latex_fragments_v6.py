#!/usr/bin/env python3
"""
Generate publication-quality LaTeX table fragments from results/v6/ CSV outputs.
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
LATEX_DIR = REPO_ROOT / "latex_fragments_v6"
LATEX_DIR.mkdir(parents=True, exist_ok=True)
V6_DIR = REPO_ROOT / "results" / "v6"

# 1. Table Char vs Word BM25
df_bm25 = pd.read_csv(V6_DIR / "char_vs_word_bm25_v6.csv")
with open(LATEX_DIR / "table_char_vs_word_bm25_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/char_vs_word_bm25_v6.csv\n")
    f.write("\\begin{table*}[t]\n\\centering\\small\n")
    f.write("\\caption{Re-evaluated Word BM25 vs.\\ Character 3--5gram BM25 across Query Regimes and OCR Conditions (Test $N=306$). Paired differences with 95\\% cluster-bootstrap CIs and Holm-corrected $p$-values.}\\label{tab:char_vs_word_bm25_v6}\n")
    f.write("\\begin{tabular}{llccccc}\n\\toprule\n")
    f.write("Query Type & Text Condition & Word BM25 MRR & Char 3--5g MRR & Diff (Char - Word) & $p_{\\text{Holm}}$ & Verdict \\\\\n\\midrule\n")
    current_q = ""
    for _, r in df_bm25.iterrows():
        q_type = str(r["Query_Type"])
        q_label = q_type.capitalize() if q_type != current_q else ""
        current_q = q_type
        cond_label = str(r["Condition"]).replace("_", " ").title()
        p_val = float(r["Holm_P_Value"])
        p_str = "$<10^{-4}$" if p_val < 1e-4 else f"{p_val:.4f}"
        verdict = "Char Dominates" if "Char 3-5gram" in str(r["Verdict"]) else ("Word Dominates" if "Word BM25" in str(r["Verdict"]) else "Tied")
        f.write(f"{q_label} & {cond_label} & {r['Word_BM25_MRR']:.4f} & {r['Char_BM25_MRR']:.4f} & {r['Diff_Char_Minus_Word']:+.4f} [{r['Diff_CI_Lower']:+.4f}, {r['Diff_CI_Upper']:+.4f}] & {p_str} & {verdict} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# 2. Table Conformal Cascade
df_conf = pd.read_csv(V6_DIR / "conformal_splits_summary_v6.csv")
with open(LATEX_DIR / "table_conformal_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/conformal_splits_summary_v6.csv\n")
    f.write("\\begin{table}[t]\n\\centering\\small\n")
    f.write("\\caption{Conformal Cascade Calibration: Corrected Marginal Loss vs.\\ Flawed Conditional Ratio over 200 Cluster-Aware Splits ($N=2{,}754$ instances, nominal target $\\epsilon=0.05$).}\\label{tab:conformal_cascade_v6}\n")
    f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
    f.write("Risk Formulation & Mean $\\tau^*$ & Mean Test Risk & Split Cov. ($\\le 0.05$) & Mean Esc. Rate & Mean MRR \\\\\n\\midrule\n")
    for _, r in df_conf.iterrows():
        name = "Marginal Loss $L=1[\\text{not esc} \\land \\text{miss}]$" if "Marginal" in str(r["Formulation"]) else "Conditional Ratio $P(\\text{miss} \\mid \\text{not esc})$"
        cov_val = float(r["Empirical_Coverage_Fraction"]) * 100.0
        esc_val = float(r["Mean_Escalation_Rate"]) * 100.0
        f.write(f"{name} & {r['Mean_Tau']:.4f} & {r['Mean_Test_Risk']:.4f} & {cov_val:.1f}\\% & {esc_val:.1f}\\% & {r['Mean_MRR']:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# 3. Table OCR Errors
df_ocr = pd.read_csv(V6_DIR / "tesseract_error_rates_v6.csv")
with open(LATEX_DIR / "table_ocr_errors_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/tesseract_error_rates_v6.csv\n")
    f.write("\\begin{table}[t]\n\\centering\\small\n")
    f.write("\\caption{Standardized JiWER OCR Error Rates on SROIE Receipts ($N=626$) vs.\\ Flawed Legacy Line-Aligned Rates.}\\label{tab:ocr_error_rates_v6}\n")
    f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
    f.write("OCR Engine & Condition & Corrected CER & Corrected WER & Bag-of-Words WER & Legacy Flawed (CER / WER) \\\\\n\\midrule\n")
    for _, r in df_ocr.iterrows():
        f.write(f"{r['Engine']} & {r['Condition']} & {r['Document_CER']:.4f} & {r['Document_WER']:.4f} & {r['Bag_of_Words_WER']:.4f} & {r['Legacy_Reported_CER']:.4f} / {r['Legacy_Reported_WER']:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# 4. Table Dense Formulation
df_dense = pd.read_csv(V6_DIR / "dense_query_formulation_v6.csv")
with open(LATEX_DIR / "table_dense_formulation_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/dense_query_formulation_v6.csv\n")
    f.write("\\begin{table}[t]\n\\centering\\small\n")
    f.write("\\caption{Dense Bi-Encoder Sensitivity to Query Formulation on Clean Ground-Truth Receipt Text ($0\\%$ noise, Test $N=306$).}\\label{tab:dense_formulation_v6}\n")
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
    f.write("Query Formulation & MiniLM MRR & MiniLM Hit@10 & BGE-Base MRR & BGE-Base Hit@10 \\\\\n\\midrule\n")
    for _, r in df_dense.iterrows():
        f.write(f"{r['Query_Formulation']} & {r['MiniLM_MRR']:.4f} & {r['MiniLM_Hit10']:.4f} & {r['BGE_Base_MRR']:.4f} & {r['BGE_Base_Hit10']:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# 5. Table Multi-Hop
df_mh = pd.read_csv(V6_DIR / "multihop_independent_v6.csv")
with open(LATEX_DIR / "table_multihop_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/multihop_independent_v6.csv\n")
    f.write("\\begin{table}[t]\n\\centering\\small\n")
    f.write("\\caption{Independent Extractive PRF vs.\\ Generative LLM Multi-Hop Bridging ($N=50$ Diagnostic Queries).}\\label{tab:multihop_v6}\n")
    f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
    f.write("Comparison & Extractive Hit@10 & LLM Hit@10 & Extr Wins & LLM Wins & Exact McNemar $p$ & Verdict \\\\\n\\midrule\n")
    for _, r in df_mh.iterrows():
        f.write(f"Extractive vs.\\ LLM Bridging & {r['Extractive_Hit10']:.3f} & {r['LLM_Hit10']:.3f} & {r['Extractive_Wins']} & {r['LLM_Wins']} & {r['Exact_McNemar_P']:.4f} & {r['Verdict']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# 6. Table CLIP Evaluation
df_clip = pd.read_csv(V6_DIR / "clip_receipt_eval_v6.csv")
with open(LATEX_DIR / "table_clip_v6.tex", "w", encoding="utf-8") as f:
    f.write("% Source: results/v6/clip_receipt_eval_v6.csv\n")
    f.write("\\begin{table}[t]\n\\centering\\small\n")
    f.write("\\caption{Genuine CLIP ViT-B/32 Visual Retrieval on SROIE Document Images ($N=813$ items, Test $N=306$ probes).}\\label{tab:clip_v6}\n")
    f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
    f.write("Query Type & MRR [95\\% CI] & Hit@10 [95\\% CI] & Mean Rank & Min / Max Rank & Softmax Entropy (nats) \\\\\n\\midrule\n")
    for _, r in df_clip.iterrows():
        mrr_str = f"{r['MRR']:.4f} [{r['MRR_CI_Lower']:.4f}, {r['MRR_CI_Upper']:.4f}]"
        h10_str = f"{r['Hit10']:.4f} [{r['Hit10_CI_Lower']:.4f}, {r['Hit10_CI_Upper']:.4f}]"
        f.write(f"{r['Query_Type'].capitalize()} & {mrr_str} & {h10_str} & {r['Mean_Rank']:.1f} & {r['Min_Rank']} / {r['Max_Rank']} & {r['Mean_Softmax_Entropy_nats']:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

print("Generated all 6 LaTeX table fragments in latex_fragments_v6/ successfully.")
