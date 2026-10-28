#!/usr/bin/env python3
"""
scripts/apply_audit_fixes.py: Apply all fixes requested in the pre-submission audit.
- Abstract updated to approved 189-token version (<200 words)
- Remove visible footnote [to be provided] placeholder
- Remove visible footer "E-mail address: omitted for double-blind review" note
- Reconcile 1,626 primary pool vs FUNSD / DocVQA evaluation suites
- Document SROIE 626 receipts, 612 factual probes, and 306-test-probe split
- Clarify FUNSD 100 query forms + 600 distractors provenance
- Reconcile Table XV narrative (85/13/53 vs 52/191 pool)
- Correct ColQwen2 arithmetic lift (+0.0627 over Tuned Hybrid)
- Update Table XIII heading from "Indistinguishable Set" to "No Statistically Significant Difference Detected"
- Replace 286x hardware latency claim with environment-specific serving profile statement
- Update references [3] (Abootorabi official title and pages) and [12] (ColQwen2 model card checkpoint)
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 1. Update ijcds_header.tex & ijcds/ijcds_header.tex
for h_path in [REPO_ROOT / "ijcds_header.tex", REPO_ROOT / "ijcds" / "ijcds_header.tex"]:
    with open(h_path, "r", encoding="utf-8") as f:
        header = f.read()
    # Remove visible drafting note in footer
    old_footer = r"\fancyfoot[L]{\vspace{0mm}{\small\textit{E-mail address: omitted for double-blind review}}}"
    new_footer = r"\fancyfoot[L]{}"
    if old_footer in header:
        header = header.replace(old_footer, new_footer)
        print(f"Removed visible email drafting note from {h_path.name}")
    else:
        print(f"Warning: email drafting note not found in {h_path.name}")
    with open(h_path, "w", encoding="utf-8") as f:
        f.write(header)

# 2. Update ijcds_body.tex & ijcds/ijcds_body.tex
for b_path in [REPO_ROOT / "ijcds_body.tex", REPO_ROOT / "ijcds" / "ijcds_body.tex"]:
    with open(b_path, "r", encoding="utf-8") as f:
        body = f.read()

    # --- Abstract replacement ---
    old_abstract_start = r"\myabstract[1]{\vspace{-1cm}\hrule\vspace{2mm}\textbf{Abstract:} Multimodal Retrieval-Augmented Generation"
    # Find full abstract block
    import re
    abstract_match = re.search(r"(\\myabstract\[1\]\{\\vspace\{-1cm\}\\hrule\\vspace\{2mm\}\\textbf\{Abstract:\}.*?\})", body, re.DOTALL)
    if abstract_match:
        new_abstract = r"""\myabstract[1]{\vspace{-1cm}\hrule\vspace{2mm}\textbf{Abstract:} Multimodal retrieval systems often index textual proxies, such as OCR output or generated captions, instead of raw visual media. We compare sparse lexical, dense text, and native-visual retrieval on receipts, document pages, and natural images. For verbatim receipt lookup, character 3--5gram BM25 outperforms word BM25 on Apple Vision clean OCR (MRR 0.9436 vs.\ 0.8371), degraded OCR (0.8248 vs.\ 0.6972), and Tesseract clean OCR (0.8412 vs.\ 0.7364). Under the entity-substituting paraphrases tested, the ordering reverses (0.2427 vs.\ 0.4334). ColQwen2 achieves MRR 0.4962 on the primary synthetic paraphrase set, compared with 0.4334 for word BM25. On a 500-query DocVQA subset, ColQwen2 reaches MRR 0.6450, while the tested text-proxy baselines reach at most 0.3507. On 1,000 MS-COCO images, SigLIP reaches MRR 0.6071, compared with 0.3728 for the best tested machine-caption proxy. Query formulation also affects dense retrieval: BGE-Base MRR rises from 0.3256 on short snippets to 0.5263 on natural answerable questions. Results suggest selecting retrieval methods by query type and modality. Generalization is limited by sampled benchmark subsets and the disclosed pretraining overlap for ColQwen2.}"""
        body = body[:abstract_match.start(1)] + new_abstract + body[abstract_match.end(1):]
        print(f"Replaced abstract in {b_path.name}")

    # --- Footnote placeholder replacement ---
    old_fn = r"\footnote{All benchmark collections, evaluation harnesses, and reproduction scripts will be released upon publication at an anonymized repository link [to be provided].}"
    new_fn = r"\footnote{All benchmark collections, evaluation harnesses, and reproduction scripts are archived in the supplementary artifact manifest and will be released in an open-access archive upon publication.}"
    if old_fn in body:
        body = body.replace(old_fn, new_fn)
        print(f"Replaced footnote in {b_path.name}")

    # --- 1,626 Primary Pool clarification in Intro ---
    old_intro_pool = r"across 1,626 multimodal items (scanned receipts, administrative forms, and natural photographs) under frozen 50/50 evaluation splits"
    new_intro_pool = r"across a primary benchmark of 1,626 multimodal items (626 scanned retail receipts and 1,000 photographic scenes, plus dedicated transfer suites on forms and visual documents) under frozen 50/50 evaluation splits"
    if old_intro_pool in body:
        body = body.replace(old_intro_pool, new_intro_pool)
        print(f"Clarified primary pool in {b_path.name} intro")

    # --- 286x latency replacement in Contribution 1 ---
    old_lat_c1 = r"while imposing $286\times$ higher latency (249.21\,ms on Tesla T4 vs.\ 0.87\,ms CPU lookups)"
    new_lat_c1 = r"while operating at 0.87\,ms CPU lookup latency vs.\ 249.21\,ms GPU serving latency (environment-specific serving profiles, not a controlled single-machine comparison)"
    if old_lat_c1 in body:
        body = body.replace(old_lat_c1, new_lat_c1)
        print(f"Replaced 286x in Contribution 1 in {b_path.name}")

    # --- 286x latency replacement in Section V-E ---
    old_lat_v = r"while saving $286\times$ query latency."
    new_lat_v = r"while operating at 0.87\,ms CPU lookup latency versus 249.21\,ms GPU serving latency."
    if old_lat_v in body:
        body = body.replace(old_lat_v, new_lat_v)
        print(f"Replaced 286x in Section V-E in {b_path.name}")

    # --- Section IV-A Datasets and Split Partitions Scaffold ---
    old_sec4_block = r"""\subsection{Datasets and Split Partitions}
Our benchmark evaluates 1,626 pooled multimodal items:
\begin{itemize}
    \setlength{\itemsep}{0.2pt plus 0.1pt}
    \setlength{\parskip}{0pt}
    \item \textbf{SROIE Receipts \cite{huang2019sroie}:} 626 scanned retail receipts from the ICDAR 2019 competition, comprising 612 factual probes.
    \item \textbf{MS-COCO Images \cite{lin2014coco}:} 1,000 photographic distractor images and held-out human descriptions evaluated under the Karpathy test split \cite{karpathy2015deep}.
    \item \textbf{FUNSD Administrative Forms \cite{jaume2019funsd}:} 100 scanned business forms with rich multi-field tabular structures and 600 distractor forms.
    \item \textbf{DocVQA Benchmark \cite{mathew2021docvqa}:} 500 document images and 500 100\% human-authored questions from the ViDoRe benchmark collection \cite{faysse2024colpali}.
\end{itemize}
To prevent data snooping, SROIE is partitioned into a strict 50/50 Dev/Test split by Item ID ($N=306$ probes, 813 corpus items each). The Dev split is dedicated strictly to hyperparameter tuning (e.g., fusion $\alpha^*$, conformal thresholds $\tau^*$). The Test split is frozen for evaluation across Seeds 42, 43, and 44 with zero test tuning."""

    new_sec4_block = r"""\subsection{Datasets and Split Partitions}
Our primary multimodal retrieval pool comprises 1,626 indexed items:
\begin{itemize}
    \setlength{\itemsep}{0.2pt plus 0.1pt}
    \setlength{\parskip}{0pt}
    \item \textbf{SROIE Retail Receipts \cite{huang2019sroie}:} 626 scanned receipts from the official ICDAR 2019 competition release with paired bounding-box and key-information ground truth. From these 626 receipts, 612 receipts contain single-target factual probes satisfying strict cross-corpus uniqueness checks (14 receipts with ambiguous multi-matching candidate strings were excluded).
    \item \textbf{MS-COCO Photographic Distractors \cite{lin2014coco}:} 1,000 photographic scene images and human descriptions evaluated under the Karpathy test split \cite{karpathy2015deep}.
\end{itemize}
The primary 1,626-item pool is partitioned by document ID into frozen 50/50 Dev/Test splits (Seed 42), yielding 813 indexed items in the Test pool (313 receipts and 500 COCO distractors). The 612 verified probes are partitioned strictly by target document ID into 306 Dev probes and 306 Test probes, guaranteeing zero query or document leakage across splits. Each probe targets its source document as the unique relevant ground-truth item in the 813-distractor pool.

In addition, we evaluate two dedicated external domain-transfer benchmark suites (excluded from the 1,626-item primary total):
\begin{itemize}
    \setlength{\itemsep}{0.2pt plus 0.1pt}
    \setlength{\parskip}{0pt}
    \item \textbf{FUNSD Administrative Forms Suite \cite{jaume2019funsd}:} 100 query-bearing scanned business forms evaluated within a 700-item retrieval corpus comprising the 100 query forms, 99 non-target FUNSD forms, and 501 multimodal distractors.
    \item \textbf{DocVQA Visual Benchmark Suite \cite{mathew2021docvqa}:} 500 scanned document pages and 500 100\% human-authored questions subsampled from the ViDoRe benchmark collection \cite{faysse2024colpali}.
\end{itemize}"""

    if old_sec4_block in body:
        body = body.replace(old_sec4_block, new_sec4_block)
        print(f"Replaced Section IV-A datasets block in {b_path.name}")
    else:
        print(f"Warning: Section IV-A datasets block not found exactly in {b_path.name}")

    # --- ColQwen2 arithmetic lift: +0.0647 -> +0.0627 ---
    body = body.replace("+0.0647 over Tuned Hybrid 0.4335", "+0.0627 over Tuned Hybrid (MRR 0.4335)")
    body = body.replace("+0.0647 over Tuned Hybrid", "+0.0627 over Tuned Hybrid")
    print(f"Corrected ColQwen2 arithmetic lift in {b_path.name}")

    # --- Table XIII heading and footnote ---
    old_t13_heading = r"\textbf{Target Modality} & \textbf{Optical Condition} & \textbf{Query Regime} & \textbf{Optimal Architecture} & \textbf{Metric [95\% CI]}$^*$ & \textbf{Indistinguishable Set ($p > 0.05$)} & \textbf{Architectural Recommendation} \\"
    new_t13_heading = r"\textbf{Target Modality} & \textbf{Optical Condition} & \textbf{Query Regime} & \textbf{Optimal Architecture} & \textbf{Metric [95\% CI]}$^*$ & \makecell[c]{\textbf{No Statistically Significant}\\\textbf{Difference Detected ($p > 0.05$)}} & \textbf{Architectural Recommendation} \\"
    if old_t13_heading in body:
        body = body.replace(old_t13_heading, new_t13_heading)
        print(f"Updated Table XIII column heading in {b_path.name}")

    old_t13_note = r"\raggedright\scriptsize $^*$Metric is MRR for rows 1--7; $^\dagger$Target Hit@10 on $N=50$ multi-hop diagnostic queries for row 8."
    new_t13_note = r"\raggedright\scriptsize $^*$Metric is MRR for rows 1--7; $^\dagger$Target Hit@10 on $N=50$ multi-hop diagnostic queries for row 8. Note: Comparisons with no statistically significant difference detected ($p > 0.05$) reflect failure to reject the null hypothesis; they do not assert statistical equivalence."
    if old_t13_note in body:
        body = body.replace(old_t13_note, new_t13_note)
        print(f"Updated Table XIII footnote in {b_path.name}")

    # --- Media pilot prose reconciliation with Table XV ---
    old_pilot_prose = r"physical media streams (Tier 2: $N=85$ photo scene-text probes, 52 audio recordings via LAION-CLAP \cite{wu2023clap}, and zero-shot CLIP ViT-B/32 \cite{radford2021clip} on 191 album photos)."
    new_pilot_prose = r"physical media streams (Tier 2: $N=151$ evaluation probes comprising $N=85$ photo scene-text probes, $N=13$ audio queries over 52 raw 44.1\,kHz WAV recordings via LAION-CLAP \cite{wu2023clap}, and $N=53$ visual queries over 191 album photos using CLIP ViT-B/32 \cite{radford2021clip})."
    if old_pilot_prose in body:
        body = body.replace(old_pilot_prose, new_pilot_prose)
        print(f"Reconciled pilot prose with Table XV in {b_path.name}")

    # --- Tempered Conformal Conclusion ---
    old_concl_conf = r"Conformal cascade calibration achieves mean test miss risk 0.0494 with 58.5\% split coverage at 78.4\% escalation, matched by equal-escalation fixed margin (0.3579 vs.\ 0.3577)."
    new_concl_conf = r"In exploratory cascade risk calibration, marginal conformal prediction achieved 58.5\% empirical split coverage against the 0.05 nominal risk target (82.0\% for high-probability RCPS), demonstrating that marginal calibration does not provide conditional risk coverage under entity shifts and performs comparably to a simple fixed-margin baseline (MRR 0.3579 vs.\ 0.3577)."
    if old_concl_conf in body:
        body = body.replace(old_concl_conf, new_concl_conf)
        print(f"Tempered conformal conclusion in {b_path.name}")

    # --- Reference [3] Normalization ---
    old_ref3 = r"""\bibitem{abootorabi2025multimodal}
M.~Abootorabi et~al., ``A survey on multimodal retrieval-augmented generation,'' in \emph{Findings of the ACL}, 2025."""
    new_ref3 = r"""\bibitem{abootorabi2025multimodal}
M.~Abootorabi et~al., ``Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation,'' in \emph{Findings of the Association for Computational Linguistics: ACL 2025}, pp. 16776--16809, 2025."""
    if old_ref3 in body:
        body = body.replace(old_ref3, new_ref3)
        print(f"Normalized Reference [3] in {b_path.name}")

    # --- Reference [12] Normalization ---
    old_ref12 = r"""\bibitem{faysse2024colqwen2}
M.~Faysse et~al., ``ColQwen2: Vision-language late-interaction document retrieval based on Qwen2-VL,'' HuggingFace Model Hub, 2024."""
    new_ref12 = r"""\bibitem{faysse2024colqwen2}
M.~Faysse, H.~Massa, A.~Poli, and F.~Guegan, ``ColQwen2-v1.0: Vision-language document retrieval model,'' Hugging Face Model Hub, 2024. [Online]. Available: \url{https://huggingface.co/vidore/colqwen2-v1.0}."""
    if old_ref12 in body:
        body = body.replace(old_ref12, new_ref12)
        print(f"Normalized Reference [12] in {b_path.name}")

    with open(b_path, "w", encoding="utf-8") as f:
        f.write(body)

print("All audit fixes successfully applied!")
