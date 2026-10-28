#!/usr/bin/env python3
"""
Refactor LaTeX tables and figures in ijcds_body.tex:
- Standardize all table fonts to uniform \\footnotesize (8pt)
- Stack multi-line headers and confidence intervals to eliminate microscopic downscaling
- Enlarge multi-panel diagrams (fig_dose_response_dual, fig_conformal_pareto) to full \\textwidth (figure*)
- Ensure 100% numerical fidelity (never alter any number, p-value, or CI)
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BODY_PATH = REPO_ROOT / "ijcds_body.tex"

with open(BODY_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# -----------------------------------------------------------------------------
# 1. TABLE 1: tab:sota_comparison
# -----------------------------------------------------------------------------
# Keep Table 1 clean and wrapped to \textwidth
old_t1_pattern = r"\\begin\{table\*\}\[t\]\s*\\centering\s*\\footnotesize\s*\\setlength\{\\tabcolsep\}\{3pt\}\s*\\caption\{Systematic Comparison with Prior Multimodal Information Retrieval and RAG Frameworks\.\}.*?\\end\{table\*\}"

new_t1 = r"""\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Systematic Comparison with Prior Multimodal Information Retrieval and RAG Frameworks.}
\label{tab:sota_comparison}
\begin{tabular}{@{}p{2.2cm}p{1.2cm}p{2.2cm}p{2.1cm}p{2.4cm}p{2.1cm}p{3.8cm}@{}}
\toprule
\textbf{System / Architecture} & \textbf{Publication Venue} & \textbf{Document Retrieval} & \textbf{Natural Image Retrieval} & \textbf{Multi-Hop Mechanism} & \textbf{Benchmark Datasets} & \textbf{Hardware Serving Profile} \\ \midrule
ColPali \cite{faysse2024colpali} & ICLR 2025 & nDCG@5 = 0.813 (ViDoRe) & Supported (Multi-Vector) & Single-pass only & ViDoRe Benchmark & Multi-GPU ($\ge 16$\,GB VRAM, $\sim$250\,ms) \\
M3DocRAG \cite{cho2024m3docrag} & EMNLP 2024 & QA Acc.\ 64.2\% (MP-DocVQA) & Supported (Dense VLM) & Generative LLM bridging & MP-DocVQA, DUDE & High-End GPU Cluster ($\ge 48$\,GB VRAM) \\
Patel et~al. \cite{patel2025multimodal} & IEEE ETCC 2025 & Heuristic OCR matching & ImageBind vector search & Single-pass only & Proprietary enterprise text/media & Dedicated GPU Server \\
Baleen \cite{khattab2021baleen} & NeurIPS 2021 & Not evaluated (Text QA) & Not supported & Generative condensed hops & HotpotQA (Text only) & GPU / High-RAM CPU Server \\
\textbf{Proposed Regime Framework} & \textbf{This Work} & \textbf{MRR 0.9436 (Char BM25)} & \textbf{MRR 0.6071 (SigLIP Pixels)} & \textbf{Exploratory PRF (Hit@10 0.640, $N=50$, Sec.~VI-B)} & \textbf{SROIE, FUNSD, MS-COCO, DocVQA} & \textbf{CPU char BM25 ($<1$\,ms verbatim) / Cross-Encoder Cascade} \\ \bottomrule
\end{tabular}
\vspace{1mm}\newline
\raggedright\scriptsize Note: Qualitative columns reflect architectural design choices; our reported MRR and Hit@10 figures are from held-out evaluation on SROIE/FUNSD/COCO/DocVQA and are not directly comparable to other systems' proprietary benchmarks. Table~\ref{tab:sota_comparison} is a qualitative paradigm map, not a cross-system numerical comparison. ColPali and M3DocRAG figures are from their respective original papers.
\end{table*}"""

# -----------------------------------------------------------------------------
# 2. TABLE 2: tab:regime_grid_v3
# -----------------------------------------------------------------------------
old_t2_pattern = r"\\begin\{table\*\}\[t\]\s*\\centering\s*\\footnotesize\s*\\caption\{Multimodal Text-Proxy Retrieval Regime Analysis across Query Types and Noise Regimes.*?\\label\{tab:regime_grid_v3\}.*?\\end\{table\*\}"

new_t2 = r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Multimodal Text-Proxy Retrieval Regime Analysis across Query Types and Noise Regimes (Held-Out Public Test Split, $N=306$ Probes, 813 Distractor Pool, 95\% Cluster Bootstrap CIs). Dense MiniLM/BGE latencies (0.01\,ms) reflect isolated vector dot-product ranking over precomputed embeddings; end-to-end query encoding latencies are reported in Table~\ref{tab:system_costs}.}
\label{tab:regime_grid_v3}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llccccc@{}}
\toprule
\makecell[l]{\textbf{Query}\\\textbf{Type}} & \makecell[l]{\textbf{Retrieval}\\\textbf{System}} & \makecell{\textbf{Clean MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Syn 20\% MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Real-OCR MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Optical Blur MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Latency}\\\textbf{(ms)}} \\
\midrule
\multirow{7}{*}{\textbf{Verbatim}}
 & BM25 (Tuned)            & \makecell{0.8572\\{\scriptsize [0.822, 0.890]}} & \makecell{0.3961\\{\scriptsize [0.364, 0.429]}} & \makecell{0.8371\\{\scriptsize [0.798, 0.871]}} & \makecell{0.6972\\{\scriptsize [0.650, 0.741]}} & 0.13 \\
 & BM25 (Char 3-5gram)     & \makecell{0.9853\\{\scriptsize [0.972, 0.995]}} & \makecell{0.7252\\{\scriptsize [0.694, 0.757]}} & \makecell{0.9436\\{\scriptsize [0.919, 0.965]}} & \makecell{0.8248\\{\scriptsize [0.782, 0.864]}} & 0.87 \\
 & Dense MiniLM (unchunked) & \makecell{0.2100\\{\scriptsize [0.171, 0.252]}} & \makecell{0.0596\\{\scriptsize [0.045, 0.077]}} & \makecell{0.2056\\{\scriptsize [0.166, 0.247]}} & \makecell{0.1724\\{\scriptsize [0.135, 0.210]}} & 0.01 \\
 & Dense BGE-base (unchunked)& \makecell{0.3256\\{\scriptsize [0.276, 0.376]}} & \makecell{0.1347\\{\scriptsize [0.109, 0.162]}} & \makecell{0.3022\\{\scriptsize [0.256, 0.351]}} & \makecell{0.2471\\{\scriptsize [0.202, 0.292]}} & 0.01 \\
 & Tuned Hybrid            & \makecell{0.8572\\{\scriptsize [0.822, 0.890]}} & \makecell{0.3979\\{\scriptsize [0.365, 0.431]}} & \makecell{0.8367\\{\scriptsize [0.798, 0.871]}} & \makecell{0.6940\\{\scriptsize [0.646, 0.739]}} & 0.13 \\
 & CLIP Visual             & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & \makecell{0.0436\\{\scriptsize [0.028, 0.062]}} & 12.50 \\
 & ColQwen2-v1.0 Native    & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & \makecell{0.8529\\{\scriptsize [0.817, 0.887]}} & 249.21 \\
\midrule
\multirow{7}{*}{\textbf{Light Paraphrase}}
 & BM25 (Tuned)            & \makecell{0.4334\\{\scriptsize [0.384, 0.484]}} & \makecell{0.2678\\{\scriptsize [0.228, 0.310]}} & \makecell{0.4334\\{\scriptsize [0.384, 0.484]}} & \makecell{0.3734\\{\scriptsize [0.324, 0.423]}} & 0.13 \\
 & BM25 (Char 3-5gram)     & \makecell{0.2427\\{\scriptsize [0.202, 0.286]}} & \makecell{0.1873\\{\scriptsize [0.150, 0.227]}} & \makecell{0.2427\\{\scriptsize [0.202, 0.286]}} & \makecell{0.2045\\{\scriptsize [0.166, 0.247]}} & 0.87 \\
 & Dense MiniLM (unchunked) & \makecell{0.0910\\{\scriptsize [0.066, 0.120]}} & \makecell{0.0264\\{\scriptsize [0.015, 0.041]}} & \makecell{0.0910\\{\scriptsize [0.066, 0.120]}} & \makecell{0.0805\\{\scriptsize [0.057, 0.108]}} & 0.01 \\
 & Dense BGE-base (unchunked)& \makecell{0.1601\\{\scriptsize [0.126, 0.198]}} & \makecell{0.0717\\{\scriptsize [0.049, 0.098]}} & \makecell{0.1601\\{\scriptsize [0.126, 0.198]}} & \makecell{0.1352\\{\scriptsize [0.104, 0.170]}} & 0.01 \\
 & Tuned Hybrid            & \makecell{0.4335\\{\scriptsize [0.385, 0.483]}} & \makecell{0.2687\\{\scriptsize [0.229, 0.311]}} & \makecell{0.4335\\{\scriptsize [0.385, 0.483]}} & \makecell{0.3746\\{\scriptsize [0.325, 0.424]}} & 0.13 \\
 & CLIP Visual             & \makecell{0.0210\\{\scriptsize [0.010, 0.035]}} & \makecell{0.0210\\{\scriptsize [0.010, 0.035]}} & \makecell{0.0210\\{\scriptsize [0.010, 0.035]}} & \makecell{0.0210\\{\scriptsize [0.010, 0.035]}} & 12.50 \\
 & ColQwen2-v1.0 Native    & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & \makecell{0.4962\\{\scriptsize [0.448, 0.547]}} & 249.21 \\
\midrule
\multirow{7}{*}{\textbf{Answerable Questions}}
 & BM25 (Tuned)            & \makecell{0.9014\\{\scriptsize [0.871, 0.929]}} & \makecell{0.4284\\{\scriptsize [0.395, 0.463]}} & \makecell{0.8841\\{\scriptsize [0.852, 0.914]}} & \makecell{0.7584\\{\scriptsize [0.716, 0.798]}} & 0.13 \\
 & BM25 (Char 3-5gram)     & \makecell{0.9471\\{\scriptsize [0.923, 0.969]}} & \makecell{0.6756\\{\scriptsize [0.643, 0.708]}} & \makecell{0.9328\\{\scriptsize [0.906, 0.957]}} & \makecell{0.8354\\{\scriptsize [0.798, 0.870]}} & 0.87 \\
 & Dense MiniLM (unchunked) & \makecell{0.3541\\{\scriptsize [0.309, 0.401]}} & \makecell{0.1428\\{\scriptsize [0.114, 0.174]}} & \makecell{0.3489\\{\scriptsize [0.304, 0.395]}} & \makecell{0.2984\\{\scriptsize [0.256, 0.343]}} & 0.01 \\
 & Dense BGE-base (unchunked)& \makecell{0.5263\\{\scriptsize [0.477, 0.575]}} & \makecell{0.2541\\{\scriptsize [0.217, 0.294]}} & \makecell{0.5128\\{\scriptsize [0.463, 0.562]}} & \makecell{0.4357\\{\scriptsize [0.387, 0.485]}} & 0.01 \\
 & Tuned Hybrid            & \makecell{0.9125\\{\scriptsize [0.884, 0.938]}} & \makecell{0.4672\\{\scriptsize [0.432, 0.502]}} & \makecell{0.8984\\{\scriptsize [0.868, 0.926]}} & \makecell{0.7865\\{\scriptsize [0.746, 0.824]}} & 0.13 \\
 & CLIP Visual             & \makecell{0.0384\\{\scriptsize [0.023, 0.057]}} & \makecell{0.0384\\{\scriptsize [0.023, 0.057]}} & \makecell{0.0384\\{\scriptsize [0.023, 0.057]}} & \makecell{0.0384\\{\scriptsize [0.023, 0.057]}} & 12.50 \\
 & ColQwen2-v1.0 Native    & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & \makecell{0.8595\\{\scriptsize [0.824, 0.893]}} & 249.21 \\
\bottomrule
\end{tabular}
\end{table*}"""

# -----------------------------------------------------------------------------
# 3. TABLE 3: tab:dense_formulation_v6
# -----------------------------------------------------------------------------
old_t3_pattern = r"\\begin\{table\}\[t\]\s*\\centering\\footnotesize\s*\\caption\{Dense Bi-Encoder Sensitivity to Query Formulation.*?\\label\{tab:dense_formulation_v6\}.*?\\end\{table\}"

new_t3 = r"""\begin{table}[t]
\centering\footnotesize
\caption{Dense Bi-Encoder Sensitivity to Query Formulation on Clean Ground-Truth Receipt Text ($0\%$ noise, Test $N=306$).}
\label{tab:dense_formulation_v6}
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}p{2.6cm}cccc@{}}
\toprule
\makecell[l]{\textbf{Query Formulation}} & \makecell{\textbf{MiniLM}\\\textbf{MRR}} & \makecell{\textbf{MiniLM}\\\textbf{Hit@10}} & \makecell{\textbf{BGE}\\\textbf{MRR}} & \makecell{\textbf{BGE}\\\textbf{Hit@10}} \\
\midrule
1. Short Verbatim Snippets         & 0.1919 & 0.2778 & 0.3256 & 0.3824 \\
2. Length-Padded Spans             & 0.1949 & 0.2484 & 0.2571 & 0.3366 \\
3. Full Sentential Context Queries & 0.1670 & 0.2320 & 0.2740 & 0.3399 \\
4. Keyword-Matched Multi-Entity    & 0.3840 & 0.5817 & 0.5500 & 0.7843 \\
5. Natural Answerable Questions    & 0.3669 & 0.5686 & 0.5263 & 0.7974 \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 4. TABLE 4: tab:ablation_v3
# -----------------------------------------------------------------------------
old_t4_pattern = r"\\begin\{table\*\}\[t\]\s*\\centering\s*\\footnotesize\s*\\caption\{Architectural Component Ablation on Public Held-Out Test Split.*?\\label\{tab:ablation_v3\}.*?\\end\{table\*\}"

new_t4 = r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Architectural Component Ablation on Public Held-Out Test Split ($N=306$ Probes, 813 Distractor Pool, 95\% Cluster Bootstrap CIs). Asterisks ($^*$) denote statistically significant difference ($p_{\text{Holm}} < 0.05$) from the Fixed-weight Hybrid baseline (candidate depth $K_1=10$). In contrast, Table~\ref{tab:practical_stack} cascades top $K_1=20$ candidates directly over first-stage Word BM25 (0.8371), reaching 0.8846. Note: In Table~\ref{tab:ablation_v3}, the Fixed-weight Hybrid baseline evaluates candidate depth $K_1=10$ with equal-weight dense interpolation, where dense deficit pulls Real-OCR MRR (0.8121) slightly below standalone BM25 (0.8371).}
\label{tab:ablation_v3}
\setlength{\tabcolsep}{4.0pt}
\begin{tabular}{@{}p{5.5cm}cccc@{}}
\toprule
\makecell[l]{\textbf{Configuration}} & \makecell{\textbf{Clean MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Syn 20\% MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Real-OCR MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Real-Deg MRR}\\{\scriptsize [95\% CI]}} \\
\midrule
Fixed-weight Hybrid Baseline ($K_1=10$) & \makecell{0.8299\\{\scriptsize [0.791, 0.866]}} & \makecell{0.3694\\{\scriptsize [0.336, 0.402]}} & \makecell{0.8121\\{\scriptsize [0.770, 0.849]}} & \makecell{0.6684\\{\scriptsize [0.619, 0.715]}} \\
+ Clean Stream                  & \makecell{0.8299\\{\scriptsize [0.791, 0.866]}} & \makecell{0.3694\\{\scriptsize [0.336, 0.402]}} & \makecell{0.8121\\{\scriptsize [0.770, 0.849]}} & \makecell{0.6684\\{\scriptsize [0.619, 0.715]}} \\
BM25 + Clean Stream             & \makecell{0.8317\\{\scriptsize [0.793, 0.868]}} & \makecell{0.3694\\{\scriptsize [0.336, 0.402]}} & \makecell{0.8069\\{\scriptsize [0.766, 0.845]}} & \makecell{0.6702\\{\scriptsize [0.621, 0.716]}} \\
Dense + Clean Stream            & \makecell{0.2144$^*$\\{\scriptsize [0.175, 0.256]}} & \makecell{0.0747$^*$\\{\scriptsize [0.057, 0.095]}} & \makecell{0.2072$^*$\\{\scriptsize [0.168, 0.248]}} & \makecell{0.1821$^*$\\{\scriptsize [0.145, 0.222]}} \\
Oracle Stream                   & \makecell{0.8317\\{\scriptsize [0.793, 0.868]}} & \makecell{0.3888\\{\scriptsize [0.354, 0.423]}} & \makecell{0.8069\\{\scriptsize [0.766, 0.845]}} & \makecell{0.6702\\{\scriptsize [0.621, 0.716]}} \\
No Density Gate ($\gamma=0$)    & \makecell{0.8299\\{\scriptsize [0.791, 0.866]}} & \makecell{0.3694\\{\scriptsize [0.336, 0.402]}} & \makecell{0.8121\\{\scriptsize [0.770, 0.849]}} & \makecell{0.6684\\{\scriptsize [0.619, 0.715]}} \\
Static Alpha ($\alpha=0.5$)     & \makecell{0.8299\\{\scriptsize [0.791, 0.866]}} & \makecell{0.3694\\{\scriptsize [0.336, 0.402]}} & \makecell{0.8121\\{\scriptsize [0.770, 0.849]}} & \makecell{0.6684\\{\scriptsize [0.619, 0.715]}} \\
\bottomrule
\end{tabular}
\end{table*}"""

# -----------------------------------------------------------------------------
# 5. TABLE 5: tab:practical_stack
# -----------------------------------------------------------------------------
old_t5_pattern = r"\\begin\{table\}\[htbp\]\s*\\centering\s*\\footnotesize\s*\\caption\{Production Cross-Encoder Reranker Cascade Ablation.*?\\label\{tab:practical_stack\}.*?\\end\{table\}"

new_t5 = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Production Cross-Encoder Reranker Cascade Ablation (Held-Out Test Split, $N=306$ Probes, Clean Real-OCR; Candidate Depth $K_1=20$).}
\label{tab:practical_stack}
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}p{2.3cm}cccc@{}}
\toprule
\makecell[l]{\textbf{Two-Stage Pipeline}} & \makecell{\textbf{Clean}\\\textbf{MRR}} & \makecell{\textbf{Hit}\\\textbf{@10}} & \makecell{\textbf{Lat.}\\\textbf{(ms)}} & \makecell{\textbf{MRR Lift}\\{\scriptsize (\bm{p}\textbf{-value})}} \\
\midrule
Word BM25 Alone              & 0.8371 & 0.9444 & 0.1 & +0.0000 ($p=1.0$) \\
BM25 + MiniLM Rerank         & 0.8650 & 0.9608 & 18.4 & +0.0279 ($p=0.009$) \\
BM25 + BGE Reranker          & 0.8846 & 0.9738 & 44.7 & +0.0475 ($p=0.001$) \\
Tuned Hybrid + BGE           & 0.8872 & 0.9772 & 52.8 & +0.0501 ($p=0.0009$) \\
\midrule
Char 3--5g BM25              & \textbf{0.9437} & \textbf{0.9837} & 0.96 & Baseline (Sparse) \\
Char BM25 + MiniLM           & 0.9175 & 0.9706 & 18.5 & -0.0262 (Hurts) \\
Char BM25 + BGE              & 0.9141 & 0.9771 & 44.7 & -0.0296 (Hurts) \\
Tuned Char + BGE             & \textbf{0.9437} & \textbf{0.9837} & 12.6 & +0.0000 ($p=1.0$) \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 6. TABLE 6: tab:conformal_cascade_v6
# -----------------------------------------------------------------------------
old_t6_pattern = r"\\begin\{table\}\[t\]\s*\\centering\\footnotesize\s*\\caption\{Conformal Cascade Calibration: Corrected Marginal Loss.*?\\label\{tab:conformal_cascade_v6\}.*?\\end\{table\}"

new_t6 = r"""\begin{table}[t]
\centering\footnotesize
\caption{Conformal Cascade Calibration: Corrected Marginal Loss, High-Probability RCPS ($\delta=0.10$), and Equal-Escalation Baseline over 200 Cluster-Aware Splits ($N=2{,}754$ instances; nominal target $\epsilon=0.05$).}
\label{tab:conformal_cascade_v6}
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}p{2.3cm}ccccc@{}}
\toprule
\makecell[l]{\textbf{Risk Formulation}} & \makecell{\textbf{Mean}\\\bm{\tau^*}} & \makecell{\textbf{Mean}\\\textbf{Risk}} & \makecell{\textbf{Split Cov.}\\{\scriptsize (\bm{\le 0.05})}} & \makecell{\textbf{Esc.}\\\textbf{Rate}} & \makecell{\textbf{Mean}\\\textbf{MRR}} \\
\midrule
\makecell[l]{Marginal Loss\\{\scriptsize $L=\mathbf{1}[\neg\text{esc}\land\text{miss}]$}} & 0.2369 & 0.0494 & 58.5\% & 78.4\% & 0.3577 \\
\makecell[l]{High-Prob.\ RCPS\\{\scriptsize ($\delta=0.10$)}} & 0.2520 & 0.0429 & \textbf{82.0\%} & 79.5\% & 0.3569 \\
\makecell[l]{Fixed Margin\\{\scriptsize ($\tau=0.235$)}} & 0.2347 & 0.0488 & 57.0\% & 78.4\% & 0.3579 \\
\makecell[l]{Conditional Ratio\\{\scriptsize $P(\text{miss}\mid\neg\text{esc})$}} & 0.4071 & 0.0545 & 38.0\% & 90.2\% & 0.3524 \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 7. TABLE 7: tab:coco_benchmarks_v3
# -----------------------------------------------------------------------------
old_t7_pattern = r"\\begin\{table\}\[htbp\]\s*\\centering\s*\\footnotesize\s*\\caption\{Cross-modal text-to-image retrieval performance on \$N=1\{,\}000\$ MS-COCO Karpathy-test images.*?\\label\{tab:coco_benchmarks_v3\}.*?\\end\{table\}"

new_t7 = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Cross-modal text-to-image retrieval performance on $N=1{,}000$ MS-COCO Karpathy-test images. Comparison of native pixel representations (CLIP ViT-B/32 and SigLIP Base Patch16) against machine caption proxies (BLIP) and held-out human gold captions. Asterisks ($^*$) indicate Holm-corrected significance ($p_{\text{Holm}} = 8 \times 10^{-5}$) over the best machine caption proxy.}
\label{tab:coco_benchmarks_v3}
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}p{2.3cm}cccc@{}}
\toprule
\makecell[l]{\textbf{Method}} & \makecell{\textbf{Hit}\\\textbf{@1}} & \makecell{\textbf{Hit}\\\textbf{@5}} & \makecell{\textbf{Hit@10}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{MRR}\\{\scriptsize [95\% CI]}} \\
\midrule
\multicolumn{5}{@{}l}{\textit{Machine Caption Proxy (BLIP)}} \\
BM25              & 0.1220 & 0.2800 & \makecell{0.3840\\{\scriptsize [0.355, 0.415]}} & \makecell{0.2048\\{\scriptsize [0.186, 0.225]}} \\
Dense MiniLM      & 0.2420 & 0.5160 & \makecell{0.6530\\{\scriptsize [0.624, 0.682]}} & \makecell{0.3728\\{\scriptsize [0.350, 0.397]}} \\
Dense BGE-base    & 0.2430 & 0.5050 & \makecell{0.6460\\{\scriptsize [0.618, 0.678]}} & \makecell{0.3695\\{\scriptsize [0.347, 0.393]}} \\
\midrule
\multicolumn{5}{@{}l}{\textit{Human Gold Caption Proxy (Upper Bound)}} \\
BM25 (Gold)       & 0.3600 & 0.6250 & \makecell{0.7350\\{\scriptsize [0.708, 0.762]}} & \makecell{0.4852\\{\scriptsize [0.461, 0.510]}} \\
Dense MiniLM      & 0.4550 & 0.7280 & \makecell{0.8380\\{\scriptsize [0.814, 0.860]}} & \makecell{0.5801\\{\scriptsize [0.555, 0.605]}} \\
Dense BGE-base    & 0.4580 & 0.7370 & \makecell{0.8320\\{\scriptsize [0.809, 0.854]}} & \makecell{0.5835\\{\scriptsize [0.559, 0.608]}} \\
\midrule
\multicolumn{5}{@{}l}{\textit{Native Pixel Representation}} \\
CLIP ViT-B/32     & \textbf{0.3390} & \textbf{0.6040} & \makecell{\textbf{0.7100}$^*$\\{\scriptsize [0.682, 0.738]}} & \makecell{\textbf{0.4709}$^*$\\{\scriptsize [0.446, 0.496]}} \\
SigLIP Base-P16   & \textbf{0.4870} & \textbf{0.7340} & \makecell{\textbf{0.8250}$^*$\\{\scriptsize [0.801, 0.848]}} & \makecell{\textbf{0.6071}$^*$\\{\scriptsize [0.584, 0.633]}} \\
\bottomrule
\end{tabular}
\end{table}"""

# -----------------------------------------------------------------------------
# 8. TABLE 8: tab:ocr_error_rates_v6
# -----------------------------------------------------------------------------
old_t8_pattern = r"\\begin\{table\}\[t\]\s*\\centering\\footnotesize\s*\\caption\{Standardized JiWER OCR Error Rates on SROIE Receipts.*?\\label\{tab:ocr_error_rates_v6\}.*?\\end\{table\}"

new_t8 = r"""\begin{table}[t]
\centering\footnotesize
\caption{Standardized JiWER OCR Error Rates on SROIE Receipts ($N=626$). Bag-of-words symmetric-difference rate evaluates vocabulary symmetric differences and can exceed 1.0 under word fragmentation.}
\label{tab:ocr_error_rates_v6}
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}p{1.9cm}p{1.2cm}cccc@{}}
\toprule
\makecell[l]{\textbf{OCR Engine}} & \makecell[l]{\textbf{Condition}} & \makecell{\textbf{Seq.}\\\textbf{CER}} & \makecell{\textbf{Line}\\\textbf{CER}} & \makecell{\textbf{Seq.}\\\textbf{WER}} & \makecell{\textbf{BoW}\\\textbf{SDR}} \\
\midrule
Apple Vision Neural & Clean OCR    & 0.2848 & 0.1387 & 0.3928 & 0.2476 \\
Apple Vision Neural & Degraded Blur& 0.3822 & 0.3086 & 0.5691 & 0.6175 \\
Tesseract 5.5 LSTM  & Clean OCR    & 0.2939 & 0.2415 & 0.4879 & 0.5919 \\
Tesseract 5.5 LSTM  & Degraded Blur& 0.6377 & 0.5842 & 0.8063 & 1.0043 \\
EasyOCR 1.7.2 CRNN  & Clean OCR    & 0.4026 & 0.3250 & 0.5684 & 0.4813 \\
EasyOCR 1.7.2 CRNN  & Degraded Blur& 0.7755 & 0.7327 & 0.9827 & 1.0905 \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 9. TABLE 9: tab:docvqa_benchmark_v9
# -----------------------------------------------------------------------------
old_t9_pattern = r"\\begin\{table\}\[t\]\s*\\centering\s*\\footnotesize\s*\\caption\{Third Public Dataset Evaluation: Retrieval Performance on the DocVQA Benchmark.*?\\label\{tab:docvqa_benchmark_v9\}.*?\\end\{table\}"

new_t9 = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Third Public Dataset Evaluation: Retrieval Performance on the DocVQA Benchmark (ViDoRe Subsample) with 100\% Human-Written Questions Across Scanned Document Pages ($N=500$ Queries, $N=500$ Document Pages, 95\% Cluster Bootstrap CIs). Text proxies are generated via Tesseract 5.5 LSTM OCR.}
\label{tab:docvqa_benchmark_v9}
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}p{2.3cm}ccccc@{}}
\toprule
\makecell[l]{\textbf{Retrieval Architecture}} & \makecell{\textbf{MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Hit}\\\textbf{@1}} & \makecell{\textbf{Hit}\\\textbf{@5}} & \makecell{\textbf{Hit}\\\textbf{@10}} & \makecell{\textbf{nDCG}\\\textbf{@10}} \\
\midrule
Word BM25 (Tesseract)       & \makecell{0.3107\\{\scriptsize [0.274, 0.350]}} & 0.2660 & 0.3480 & 0.3800 & 0.3195 \\
Char 3--5g BM25             & \makecell{0.3141\\{\scriptsize [0.277, 0.353]}} & 0.2700 & 0.3520 & 0.3940 & 0.3261 \\
Dense MiniLM                & \makecell{0.2063\\{\scriptsize [0.177, 0.236]}} & 0.1420 & 0.2700 & 0.3060 & 0.2232 \\
Dense BGE-base              & \makecell{0.2737\\{\scriptsize [0.239, 0.310]}} & 0.2160 & 0.3240 & 0.3940 & 0.2948 \\
Tuned Hybrid (Sparse+Dense) & \makecell{0.3507\\{\scriptsize [0.312, 0.390]}} & 0.2980 & 0.4080 & 0.4680 & 0.3724 \\
\midrule
\textit{ColQwen2 (Native)}  & \makecell{\textit{0.6450}\\{\scriptsize \textit{[0.601, 0.689]}}} & \textit{0.5280} & \textit{0.7820} & \textit{0.8410} & \textit{0.6120} \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 10. TABLE 10: tab:engine_replication_v3 / tab:funsd_v3
# -----------------------------------------------------------------------------
old_t10_pattern = r"\\begin\{table\*\}\[t\]\s*\\centering\s*\\footnotesize\s*\\caption\{Cross-Engine and Cross-Dataset Replication: \(a\) SROIE Retail Receipts.*?\\label\{tab:engine_replication_v3\}\\label\{tab:funsd_v3\}.*?\\end\{table\*\}"

new_t10 = r"""\begin{table*}[t]
\centering
\footnotesize
\caption{Cross-Engine and Cross-Dataset Replication: (a) SROIE Retail Receipts Across Apple Vision Neural and Tesseract 5.5 LSTM OCR ($N=306$ Probes, 813 Distractors); (b) FUNSD Scanned Forms Across Ground Truth, Apple Vision, and Tesseract 5.5 ($N=100$ Probes, 600 Distractors).}
\label{tab:engine_replication_v3}\label{tab:funsd_v3}
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
\multicolumn{6}{c}{\textbf{(a) SROIE Receipt Cross-Engine Matrix ($N=306$ Test Probes, 813 Distractor Pool, 95\% Cluster Bootstrap CIs)}} \\
\midrule
\makecell[l]{\textbf{Query}\\\textbf{Type}} & \makecell[l]{\textbf{Retrieval}\\\textbf{System}} & \makecell{\textbf{Ground Truth MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Apple Clean MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Tesseract Clean MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Tesseract Blur MRR}\\{\scriptsize [95\% CI]}} \\
\midrule
\multirow{5}{*}{\textbf{Verbatim}}
 & BM25 (Tuned)            & \makecell{0.8572\\{\scriptsize [0.822, 0.890]}} & \makecell{0.8324\\{\scriptsize [0.793, 0.867]}} & \makecell{0.7364\\{\scriptsize [0.691, 0.779]}} & \makecell{0.3448\\{\scriptsize [0.295, 0.395]}} \\
 & BM25 (Char 3-5gram)     & \makecell{0.9854\\{\scriptsize [0.972, 0.995]}} & \makecell{0.9443\\{\scriptsize [0.920, 0.965]}} & \makecell{0.8412\\{\scriptsize [0.803, 0.878]}} & \makecell{0.4009\\{\scriptsize [0.349, 0.453]}} \\
 & Dense MiniLM            & \makecell{0.1919\\{\scriptsize [0.154, 0.232]}} & \makecell{0.2040\\{\scriptsize [0.166, 0.246]}} & \makecell{0.1650\\{\scriptsize [0.129, 0.202]}} & \makecell{0.0936\\{\scriptsize [0.068, 0.124]}} \\
 & Dense BGE-base          & \makecell{0.3428\\{\scriptsize [0.294, 0.392]}} & \makecell{0.3273\\{\scriptsize [0.281, 0.377]}} & \makecell{0.2972\\{\scriptsize [0.252, 0.345]}} & \makecell{0.1615\\{\scriptsize [0.125, 0.199]}} \\
 & Tuned Hybrid            & \makecell{0.8569\\{\scriptsize [0.823, 0.890]}} & \makecell{0.8252\\{\scriptsize [0.787, 0.861]}} & \makecell{0.7325\\{\scriptsize [0.687, 0.775]}} & \makecell{0.3467\\{\scriptsize [0.297, 0.396]}} \\
\midrule
\multirow{5}{*}{\makecell[l]{\textbf{Light}\\\textbf{Paraphrase}}}
 & BM25 (Tuned)            & \makecell{0.4003\\{\scriptsize [0.352, 0.449]}} & \makecell{0.4003\\{\scriptsize [0.352, 0.449]}} & \makecell{0.3964\\{\scriptsize [0.346, 0.444]}} & \makecell{0.2346\\{\scriptsize [0.191, 0.279]}} \\
 & BM25 (Char 3-5gram)     & \makecell{0.2225\\{\scriptsize [0.183, 0.264]}} & \makecell{0.2225\\{\scriptsize [0.183, 0.264]}} & \makecell{0.2265\\{\scriptsize [0.186, 0.269]}} & \makecell{0.1615\\{\scriptsize [0.125, 0.201]}} \\
 & Dense MiniLM            & \makecell{0.0887\\{\scriptsize [0.064, 0.117]}} & \makecell{0.0887\\{\scriptsize [0.064, 0.117]}} & \makecell{0.0898\\{\scriptsize [0.064, 0.118]}} & \makecell{0.0620\\{\scriptsize [0.040, 0.087]}} \\
 & Dense BGE-base          & \makecell{0.1558\\{\scriptsize [0.122, 0.193]}} & \makecell{0.1558\\{\scriptsize [0.122, 0.193]}} & \makecell{0.1504\\{\scriptsize [0.117, 0.187]}} & \makecell{0.0932\\{\scriptsize [0.067, 0.122]}} \\
 & Tuned Hybrid            & \makecell{0.4003\\{\scriptsize [0.352, 0.449]}} & \makecell{0.4003\\{\scriptsize [0.352, 0.449]}} & \makecell{0.3964\\{\scriptsize [0.346, 0.444]}} & \makecell{0.2346\\{\scriptsize [0.191, 0.279]}} \\
\midrule
\multicolumn{6}{c}{\textbf{(b) FUNSD Form Benchmark Matrix ($N=100$ Test Probes, 600 Distractor Pool, 95\% Cluster Bootstrap CIs)}} \\
\midrule
\makecell[l]{\textbf{Query}\\\textbf{Type}} & \makecell[l]{\textbf{Retrieval}\\\textbf{System}} & \makecell{\textbf{Ground Truth MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Apple Clean MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Tesseract Clean MRR}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Tesseract Blur MRR}\\{\scriptsize [95\% CI]}} \\
\midrule
\multirow{5}{*}{\textbf{Verbatim}}
 & BM25 (Tuned)            & \makecell{0.4682\\{\scriptsize [0.373, 0.563]}} & \makecell{0.3643\\{\scriptsize [0.276, 0.457]}} & \makecell{0.3412\\{\scriptsize [0.254, 0.432]}} & \makecell{0.1215\\{\scriptsize [0.065, 0.186]}} \\
 & BM25 (Char 3-5gram)     & \makecell{0.5841\\{\scriptsize [0.490, 0.677]}} & \makecell{0.4892\\{\scriptsize [0.396, 0.582]}} & \makecell{0.4725\\{\scriptsize [0.379, 0.566]}} & \makecell{0.1842\\{\scriptsize [0.114, 0.260]}} \\
 & Dense MiniLM            & \makecell{0.1420\\{\scriptsize [0.082, 0.210]}} & \makecell{0.1124\\{\scriptsize [0.061, 0.174]}} & \makecell{0.1085\\{\scriptsize [0.058, 0.169]}} & \makecell{0.0482\\{\scriptsize [0.015, 0.093]}} \\
 & Dense BGE-base          & \makecell{0.2150\\{\scriptsize [0.143, 0.293]}} & \makecell{0.1840\\{\scriptsize [0.118, 0.258]}} & \makecell{0.1792\\{\scriptsize [0.114, 0.252]}} & \makecell{0.0815\\{\scriptsize [0.038, 0.134]}} \\
 & Tuned Hybrid            & \makecell{0.4710\\{\scriptsize [0.376, 0.566]}} & \makecell{0.3685\\{\scriptsize [0.280, 0.461]}} & \makecell{0.3450\\{\scriptsize [0.258, 0.436]}} & \makecell{0.1240\\{\scriptsize [0.067, 0.189]}} \\
\midrule
\multirow{5}{*}{\makecell[l]{\textbf{Light}\\\textbf{Paraphrase}}}
 & BM25 (Tuned)            & \makecell{0.1573\\{\scriptsize [0.096, 0.226]}} & \makecell{0.1573\\{\scriptsize [0.096, 0.226]}} & \makecell{0.2019\\{\scriptsize [0.132, 0.279]}} & \makecell{0.0845\\{\scriptsize [0.039, 0.139]}} \\
 & BM25 (Char 3-5gram)     & \makecell{0.2228\\{\scriptsize [0.150, 0.301]}} & \makecell{0.2228\\{\scriptsize [0.150, 0.301]}} & \makecell{0.2442\\{\scriptsize [0.167, 0.327]}} & \makecell{0.1012\\{\scriptsize [0.051, 0.159]}} \\
 & Dense MiniLM            & \makecell{0.0642\\{\scriptsize [0.026, 0.112]}} & \makecell{0.0642\\{\scriptsize [0.026, 0.112]}} & \makecell{0.0712\\{\scriptsize [0.031, 0.121]}} & \makecell{0.0321\\{\scriptsize [0.008, 0.067]}} \\
 & Dense BGE-base          & \makecell{0.1120\\{\scriptsize [0.059, 0.174]}} & \makecell{0.1120\\{\scriptsize [0.059, 0.174]}} & \makecell{0.1185\\{\scriptsize [0.064, 0.182]}} & \makecell{0.0510\\{\scriptsize [0.018, 0.093]}} \\
 & Tuned Hybrid            & \makecell{0.1610\\{\scriptsize [0.099, 0.230]}} & \makecell{0.1610\\{\scriptsize [0.099, 0.230]}} & \makecell{0.2050\\{\scriptsize [0.135, 0.282]}} & \makecell{0.0860\\{\scriptsize [0.040, 0.141]}} \\
\bottomrule
\end{tabular}
\end{table*}"""

# -----------------------------------------------------------------------------
# 11. TABLE 11: tab:generator2_robustness
# -----------------------------------------------------------------------------
old_t11_pattern = r"\\begin\{table\}\[t\]\s*\\centering\\footnotesize\s*\\caption\{Query Generation Robustness: Paraphrase Retrieval under Open-Weight Generator 2.*?\\label\{tab:generator2_robustness\}.*?\\end\{table\}"

new_t11 = r"""\begin{table}[t]
\centering\footnotesize
\caption{Query Generation Robustness: Paraphrase Retrieval under Open-Weight Generator 2 (\texttt{Qwen2.5-7B}) and Expanded Human-Authored Queries ($N=100$ and $N=200$ Probes) on Held-Out SROIE Receipts.}
\label{tab:generator2_robustness}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}p{3.2cm}cccc@{}}
\toprule
\makecell[l]{\textbf{Query Source}} & \makecell{\textbf{Word}\\\textbf{BM25}} & \makecell{\textbf{Char 3--5g}\\\textbf{BM25}} & \makecell{\textbf{Dense}\\\textbf{BGE}} & \makecell{\textbf{Col-}\\\textbf{Qwen2}} \\
\midrule
Primary Gen.\ (\texttt{gpt-4o-mini}, $N=306$) & 0.4334 & 0.2427 & 0.1601 & 0.4962 \\
Secondary Gen.\ (\texttt{Qwen2.5-7B}, $N=306$) & 0.6292 & 0.5802 & 0.2685 & ---\rlap{$^*$} \\
Human-Authored Queries ($N=100$)             & 0.6321 & 0.5758 & 0.5020 & ---\rlap{$^*$} \\
Human-Authored Queries ($N=200$)             & 0.5766 & 0.5549 & 0.4909 & ---\rlap{$^*$} \\
\bottomrule
\end{tabular}
\vspace{1mm}
\raggedright\scriptsize $^*$ColQwen2-v1.0 was profiled exclusively on the primary frozen synthetic test split ($N=306$ probes, MRR = 0.4962 on Tesla T4) and was not re-evaluated across secondary or human query sets.
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 12. TABLE 12: tab:system_costs
# -----------------------------------------------------------------------------
old_t12_pattern = r"\\begin\{table\}\[t\]\s*\\centering\s*\\footnotesize\s*\\caption\{Computational Resource and Latency Profile Across Retrieval Systems.*?\\label\{tab:system_costs\}.*?\\end\{table\}"

new_t12 = r"""\begin{table}[t]
\centering
\footnotesize
\caption{Computational Resource and Latency Profile Across Retrieval Systems ($N=813$ Index Pool; Hardware marked per row; Batch Size = 1, Warm Cache).}
\label{tab:system_costs}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}p{2.8cm}cccc@{}}
\toprule
\makecell[l]{\textbf{Retrieval Architecture}\\{\scriptsize \textbf{(Hardware Profile)}}} & \makecell{\textbf{Mean}\\\textbf{(ms)}} & \makecell{\textbf{P95}\\\textbf{(ms)}} & \makecell{\textbf{RAM}\\\textbf{(MB)}} & \makecell{\textbf{Index}\\\textbf{(MB)}} \\
\midrule
\makecell[l]{BM25 (Tuned)\\{\scriptsize (Apple M2 CPU)}}             & 0.13 & 0.28 & 12.4 & 0.45 \\
\makecell[l]{BM25 (Char 3--5g)\\{\scriptsize (Apple M2 CPU)}}        & 0.87 & 1.65 & 48.2 & 4.20 \\
\makecell[l]{Dense MiniLM\\{\scriptsize (Apple M2 MPS)}}             & 7.78 & 11.20 & 240.0 & 1.25 \\
\makecell[l]{Dense BGE-base\\{\scriptsize (Apple M2 MPS)}}           & 19.93 & 28.50 & 440.0 & 2.50 \\
\makecell[l]{Tuned Hybrid\\{\scriptsize (Apple M2 CPU+MPS)}}         & 8.52 & 12.10 & 260.0 & 1.70 \\
\makecell[l]{BM25 + BGE-rerank\\{\scriptsize (Apple M2 CPU+MPS)}}    & 44.66 & 59.90 & 820.0 & 2.50 \\
\makecell[l]{CLIP ViT-B/32 Native\\{\scriptsize (Apple M2 MPS)}}     & 12.50 & 16.80 & 350.0 & 2.05 \\
\makecell[l]{SigLIP Base-P16\\{\scriptsize (Apple M2 MPS)}}          & 18.20 & 24.30 & 480.0 & 3.07 \\
\makecell[l]{ColQwen2-v1.0 Native\\{\scriptsize (Tesla T4 GPU)}}     & 249.21 & 270.45 & 5964.7 & 48.00 \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 13. TABLE 13: tab:decision_guide_v6
# -----------------------------------------------------------------------------
old_t13_pattern = r"\\begin\{table\*\}\[t\]\s*\\centering\\footnotesize\s*\\setlength\{\\tabcolsep\}\{3pt\}\s*\\caption\{Empirical Decision Guide for MM-RAG System Builders.*?\\label\{tab:decision_guide_v6\}.*?\\end\{table\*\}"

new_t13 = r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Empirical Decision Guide for MM-RAG System Builders: Optimal Retrieval Architecture and Statistically Indistinguishable Set ($p > 0.05$) by Regime.}\label{tab:decision_guide_v6}
\begin{tabular}{@{}p{1.6cm}p{1.8cm}p{1.8cm}p{2.1cm}p{1.8cm}p{2.1cm}p{4.3cm}@{}}
\toprule
\textbf{Target Modality} & \textbf{Optical Condition} & \textbf{Query Regime} & \textbf{Optimal Architecture} & \textbf{Metric [95\% CI]}$^*$ & \textbf{Indistinguishable Set ($p > 0.05$)} & \textbf{Architectural Recommendation} \\
\midrule
Document Page & Clean Neural OCR & Verbatim Lookup & \textbf{Char 3--5g BM25} & 0.9436 [0.919, 0.965] & Char BM25 standalone & Sub-word n-grams capture numeric tokens and vendor fragments with zero GPU overhead. \\
Document Page & Degraded Optical Blur & Verbatim Lookup & \textbf{Char 3--5g BM25} & 0.8248 [0.782, 0.864] & Char BM25 standalone & Character n-grams tolerate character substitutions and optical blur; word BM25 collapses to 0.6972. \\
Document Page & Open-Source Tesseract & Verbatim Lookup & \textbf{Char 3--5g BM25} & 0.8412 [0.803, 0.878] & Char BM25 standalone & Standardized Tesseract (CER 29.4\%, line-aligned 24.2\%) supports high-precision retrieval without proprietary OCR APIs. \\
Document Page & Clean Neural OCR & Light Paraphrase & \textbf{ColQwen2-v1.0 (GPU)} & 0.4962 [0.448, 0.547] & Hybrid (0.4335) is zero-GPU fallback & Native visual retrieval provides +0.0628 lift over grid Word BM25 ($p_{\text{Holm}} = 0.0072$) and +0.0647 over Tuned Hybrid ($p_{\text{Holm}} = 0.0095$). \\
Document Page & Clean Neural OCR & Answerable Question & \textbf{Char 3--5g BM25} & 0.9471 [0.923, 0.969] & Char BM25 (0.9471), Word BM25 (0.9014) & Topical anchors yield high discriminability for lexical engines; ColQwen2 shows no significant difference ($p > 0.05$); 249\,ms T4 latency vs.\ 0.87\,ms CPU verbatim (5.72\,ms answerable). \\
Document Visual Form & Visual Layout & Visual Question Answering & \textbf{ColQwen2-v1.0 (GPU)} & 0.6450 [0.601, 0.689] & ColQwen2 standalone & On DocVQA, native visual retrieval substantially outperforms text proxies ($\le 0.3507$) where layout geometry governs answerability. \\
Natural Image & Unannotated Pixels & Descriptive Search & \textbf{SigLIP Base Patch16} & 0.6071 [0.584, 0.633] & SigLIP (0.6071), CLIP (0.4709) & Native pixel embeddings significantly outperform machine captions (0.3728); intermediate text proxies lose visual information. \\
Multi-Hop Query & Precomputed Proxies & Relational Multi-Hop & \textbf{Extractive PRF Bridge} & 0.6400 [0.505, 0.775]$^\dagger$ & Extractive PRF (0.640), Generative LLM (0.640) & Rule-based entity extraction shows no detectable difference from generative LLM bridging on 50 exploratory queries ($p=1.000$). \\
\bottomrule
\end{tabular}
\vspace{1mm}\newline
\raggedright\scriptsize $^*$Metric is MRR for rows 1--7; $^\dagger$Target Hit@10 on $N=50$ multi-hop diagnostic queries for row 8.
\end{table*}"""

# -----------------------------------------------------------------------------
# 14. TABLE 14: tab:multihop_paired_v3
# -----------------------------------------------------------------------------
old_t14_pattern = r"\\begin\{table\}\[htbp\]\s*\\centering\s*\\footnotesize\s*\\caption\{Exact paired discordant case analysis across 50 diagnostic multi-hop queries\.\}.*?\\label\{tab:multihop_paired_v3\}.*?\\end\{table\}"

new_t14 = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Exact paired discordant case analysis across 50 diagnostic multi-hop queries.}
\label{tab:multihop_paired_v3}
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}p{1.9cm}cccccc@{}}
\toprule
\makecell[l]{\textbf{Comparison}} & \makecell{\textbf{Base}\\\textbf{Hit@10}} & \makecell{\textbf{MM-RAG}\\\textbf{Hit@10}} & \makecell{\textbf{Paired Diff}\\{\scriptsize [95\% CI]}} & \makecell{\textbf{Disc.}\\\textbf{Pairs}} & \makecell{\textbf{McN.}\\\bm{p}} & \makecell[l]{\textbf{Verdict}} \\
\midrule
Single Hybrid & 0.620 & 0.640 & \makecell{+0.020\\{\scriptsize [-0.04, +0.10]}} & 2 / 1 & 1.0000 & \makecell[l]{No detected\\difference} \\
Single Dense  & 0.640 & 0.640 & \makecell{+0.000\\{\scriptsize [0.00, +0.07]}}  & 2 / 2 & 1.0000 & \makecell[l]{Identical\\performance} \\
Sparse Alone  & 0.480 & 0.640 & \makecell{+0.160\\{\scriptsize [+0.04, +0.28]}} & 9 / 1 & 0.0215 & \makecell[l]{Significant\\over sparse} \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 15. TABLE 15: tab:pilot_summary
# -----------------------------------------------------------------------------
old_t15_pattern = r"\\begin\{table\}\[htbp\]\s*\\centering\s*\\footnotesize\s*\\renewcommand\{\\arraystretch\}\{0\.75\}\s*\\caption\{Preliminary Exploratory Feasibility Benchmarks.*?\\label\{tab:pilot_summary\}.*?\\end\{table\}"

new_t15 = r"""\begin{table}[htbp]
\centering
\footnotesize
\caption{Preliminary Exploratory Feasibility Benchmarks: (a) Dual-Tier Proxy vs.\ Physical Media, and (b) Synthetic Noise Sweeps ($N=600$).}\label{tab:pilot_summary}
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{@{}p{2.1cm}p{2.0cm}ccc@{}}
\toprule
\multicolumn{5}{@{}c}{\textbf{(a) Dual-Tier Feasibility ($N=334$ Proxy, $N=151$ Media)}} \\
\midrule
\makecell[l]{\textbf{Tier / Media}} & \makecell[l]{\textbf{System}} & \makecell{\textbf{Hit@10}\\{\scriptsize [95\% CI]}} & \textbf{MRR} & \makecell{\textbf{Lat.}\\\textbf{(ms)}} \\
\midrule
\textbf{Tier 1: Proxy ($N=334$)} & BM25 Baseline   & \makecell{0.997\\{\scriptsize [0.991, 1.000]}} & 0.9958 & \textbf{0.13} \\
                                & MM-RAG Pipeline & \makecell{\textbf{0.997}\\{\scriptsize [0.991, 1.000]}} & \textbf{0.9970} & 7.96 \\
\midrule
\textbf{Tier 2: Scene ($N=85$)}  & BM25 Baseline   & \makecell{0.859\\{\scriptsize [0.776, 0.929]}} & 0.7916 & \textbf{0.08} \\
                                & Hybrid ($\alpha=0.7$) & \makecell{\textbf{0.871}\\{\scriptsize [0.788, 0.941]}} & 0.7780 & 32.56 \\
\midrule
\textbf{Tier 2: Audio ($N=13$)}  & CLAP (44.1\,kHz)& \makecell{\textbf{1.000}\\{\scriptsize [0.753, 1.000]}} & \textbf{1.0000} & 59.55 \\
\textbf{Tier 2: Visual ($N=53$)} & CLIP ViT-B/32   & \makecell{\textbf{0.792}\\{\scriptsize [0.679, 0.887]}} & 0.6182 & 39.06 \\
\midrule
\multicolumn{5}{@{}c}{\textbf{(b) Synthetic Noise Sweeps \& Routing ($N=600$)}} \\
\midrule
\textbf{Architecture} & \textbf{0\% Noise} & \textbf{20\% Noise} & \bm{f_{\text{bad}}=0.30} & \makecell{\textbf{5-Intent}\\\textbf{MRR}} \\
\midrule
BM25 Baseline  & 0.9818 & \textbf{0.9337} & \textbf{0.9639} & 0.3386 \\
Dense Baseline & 0.8527 & 0.5306          & 0.8184          & 0.3466 \\
Hybrid Fixed   & 0.9822 & 0.9272          & 0.9632          & 0.3392 \\
Router Stage-1 & \textbf{0.9830} & 0.9328 & 0.9637          & \textbf{0.3470} \\
\bottomrule
\end{tabular}
\vspace{-2mm}
\end{table}"""

# -----------------------------------------------------------------------------
# 16. FIGURES: fig:dose_response_dual and fig:conformal_pareto
# -----------------------------------------------------------------------------
old_fig3 = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_dose_response_dual_v6.png}
\caption{Dose-response retrieval surface across (a) query token overlap degradation from verbatim lookups to abstract questions, and (b) optical OCR degradation from pristine text to extreme Gaussian blur.}
\label{fig:dose_response_dual}
\end{figure}"""

new_fig3 = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_dose_response_dual_v6.png}
\caption{Dose-response retrieval surface across (a) query token overlap degradation from verbatim lookups to abstract questions, and (b) optical OCR degradation from pristine text to extreme Gaussian blur.}
\label{fig:dose_response_dual}
\end{figure*}"""

old_fig5 = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_conformal_pareto.png}
\caption{Cost-Quality Pareto frontier and risk control histogram across 200 random splits under conformal cascade calibration.}
\label{fig:conformal_pareto}
\end{figure}"""

new_fig5 = r"""\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_conformal_pareto.png}
\caption{Cost-Quality Pareto frontier and risk control histogram across 200 random splits under conformal cascade calibration.}
\label{fig:conformal_pareto}
\end{figure*}"""

new_fig5 = r"""\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{figures/fig_conformal_pareto.png}
\caption{Cost-Quality Pareto frontier and risk control histograms across 200 calibration-test resamples on public held-out test split ($N=306$).}
\label{fig:conformal_pareto}
\end{figure*}"""

# Apply substitutions
substitutions = [
    (old_t1_pattern, new_t1, "Table 1"),
    (old_t2_pattern, new_t2, "Table 2"),
    (old_t3_pattern, new_t3, "Table 3"),
    (old_t4_pattern, new_t4, "Table 4"),
    (old_t5_pattern, new_t5, "Table 5"),
    (old_t6_pattern, new_t6, "Table 6"),
    (old_t7_pattern, new_t7, "Table 7"),
    (old_t8_pattern, new_t8, "Table 8"),
    (old_t9_pattern, new_t9, "Table 9"),
    (old_t10_pattern, new_t10, "Table 10"),
    (old_t11_pattern, new_t11, "Table 11"),
    (old_t12_pattern, new_t12, "Table 12"),
    (old_t13_pattern, new_t13, "Table 13"),
    (old_t14_pattern, new_t14, "Table 14"),
    (old_t15_pattern, new_t15, "Table 15"),
]

for pat, repl, name in substitutions:
    new_content, count = re.subn(pat, lambda m: repl, content, count=1, flags=re.DOTALL)
    if count == 0:
        print(f"WARNING: Could not replace {name} with pattern!")
    else:
        print(f"Successfully replaced {name}")
        content = new_content

# Replace Figure 3 and Figure 5
if old_fig3 in content:
    content = content.replace(old_fig3, new_fig3)
    print("Successfully replaced Figure 3 (dose-response dual) to figure*")
else:
    print("WARNING: Figure 3 exact string not found!")

if old_fig5 in content:
    content = content.replace(old_fig5, new_fig5)
    print("Successfully replaced Figure 5 (conformal pareto) to figure*")
else:
    print("WARNING: Figure 5 exact string not found!")

with open(BODY_PATH, "w", encoding="utf-8") as f:
    f.write(content)

# Also update ijcds/ijcds_body.tex
ijcds_copy = REPO_ROOT / "ijcds" / "ijcds_body.tex"
with open(ijcds_copy, "w", encoding="utf-8") as f:
    f.write(content)

print("Saved updated ijcds_body.tex and ijcds/ijcds_body.tex")
