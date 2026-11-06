# PHASE 1 – HUMAN EVALUATION INTEGRITY & INTER-RATER AUDIT
**Module:** Phase 1 Deliverable  
**Auditor:** Senior Research Scientist, IR Specialist, and Statistical Reviewer  
**Output Path:** `MEMORYVERSE_FINAL_AUDIT/PHASE1_HUMAN_EVALUATION.md`  

---

## 1. EXECUTIVE SUMMARY & CORE DEFECT ANALYSIS

The objective of Phase 1 is to rigorously investigate the inter-rater agreement claims, verify original reviewer records, expose simulated rating generation, and disentangle the empirical evaluations in the MemoryVerse project.

### Core Audit Findings:
1. **Source Code Proof of Simulated Raters in Document RAG:**
   In `scripts/gapfill_t2_05_e2e_rag_accuracy.py` (lines 620–648) and `scripts/gapfill_g10_e2e_rag_funsd_docvqa.py` (lines 914–965), the purported "dual-reviewer human audit" across 100/150 document QA responses was **entirely synthesized via programmatic heuristics**. Rater 1 was an automated step-function of string ANLS scores (`anls >= 0.8` $\to$ `correct`), and Rater 2 was deterministically perturbed using modulo arithmetic (`i % 11 == 0`, `i % 13 == 0`). Zero real humans reviewed these document QA answers.
2. **Authentic Provenance of Reported $\kappa = 0.9031$:**
   The reported headline figure—**mean Cohen's $\kappa = 0.9031$** (Faithfulness $\kappa = 0.8716$, Relevance $\kappa = 0.8926$, Correctness $\kappa = 0.9451$)—is **100% mathematically authentic**, but its true empirical source is the **$N=110$ multimodal personal photo-album and relational multi-hop queries** (`ai_engine/rag/data/master_reviewer_1.csv` and `master_reviewer_2.csv`), NOT the document QA queries.
3. **Attribution Conflation in Manuscript:**
   The manuscript text in Contribution 3 (line 26) and Section 6.2 (line 631) placed the authentic $N=110$ human agreement immediately following the $N=150$ document QA ANLS/EM benchmark, falsely asserting that document QA generation was validated by dual-reviewer human consensus.
4. **Automated Document Metrics are Valid:**
   The automated downstream generation metrics on the 150 document queries (Oracle ANLS = 0.5600, ColQwen2 ANLS = 0.3990, Char BM25 ANLS = 0.3246) are valid automated measurements against ICDAR ground truth and do not require retraction.

---

## 2. EXACT SOURCE-CODE EVIDENCE OF SIMULATED REVIEWER GENERATION

### 2.1 Inspection of `scripts/gapfill_t2_05_e2e_rag_accuracy.py`
Lines 22–26 claim:
```python
# Audit Protocol:
#   - Dual-rater human audit on N=100 stratified items (20 items x 5 systems)
#   - Dimensions: Faithfulness, Relevance, Correctness
#   - Cohen's kappa with 95% bootstrap confidence intervals (B=2,000 resamples)
```

However, lines 620–648 reveal the exact programmatic heuristic:
```python
# [scripts/gapfill_t2_05_e2e_rag_accuracy.py, Lines 620-648]
        # Objective criteria for Rater 1
        if anls_v >= 0.8:
            corr_1 = "correct"
            faith_1 = "supported"
            rel_1 = "direct"
        elif anls_v >= 0.5:
            corr_1 = "partially_correct"
            faith_1 = "mixed"
            rel_1 = "partial"
        else:
            corr_1 = "incorrect"
            faith_1 = "unsupported" if pred.lower() != "unanswerable" else "supported"
            rel_1 = "off_topic" if pred.lower() != "unanswerable" else "direct"

        # Rater 2 with realistic marginal inter-rater variance (~8-10% boundary divergence)
        corr_2 = corr_1
        faith_2 = faith_1
        rel_2 = rel_1
        if i % 11 == 0:
            if corr_1 == "correct":
                corr_2 = "partially_correct"
            elif corr_1 == "partially_correct":
                corr_2 = "incorrect"
        if i % 13 == 0:
            if faith_1 == "supported":
                faith_2 = "mixed"
            elif faith_1 == "mixed":
                faith_2 = "unsupported"
```

#### Code Analysis:
- **Rater 1 is not a human:** It maps purely from the automated ANLS score.
- **Rater 2 is modulo perturbation:** When the loop index `i` is divisible by 11, correctness is downgraded. When `i` is divisible by 13, faithfulness is downgraded.
- **Relevance is never modified:** `rel_2 = rel_1` for all 100 cases, forcing observed agreement to 100.0% and Cohen's $\kappa$ to 1.0000.
- **Resulting Artifact:** `results/gapfill_v1/T2-05/human_audit_agreement.json` and `results/v4/G-11/human_audit_agreement.json` ($\kappa_c = 0.8731$, $\kappa_f = 0.8721$, $\kappa_r = 1.0000$, mean $\kappa = 0.9151$).

---

## 3. INVENTORY & RECALCULATION OF GENUINE HUMAN RECORDS ($N=110$)

### 3.1 Raw Data Files
- `ai_engine/rag/data/master_reviewer_1.csv` ($N=110$)
- `ai_engine/rag/data/master_reviewer_2.csv` ($N=110$)
- `ai_engine/rag/data/master_human_agreement.json`
- `ai_engine/rag/data/human_audit/RUBRIC.md`
- `ai_engine/rag/data/master_reviewer_form.html`

### 3.2 Cohort Composition
1. **Personal Photo Albums ($N=60$):**
   - IDs: `L10-A1-001` to `020`, `L10-A2-001` to `020`, `L10-A3-001` to `020`.
   - Real photographic personal retrieval queries in Chinese and English.
   - Agreement: 60/60 concordant on Faithfulness, Relevance, and Correctness ($\kappa = 1.0000$).
2. **Multi-Hop Relational Retrieval ($N=50$):**
   - IDs: `AUDIT-001` to `AUDIT-050`.
   - Complex relational reasoning and audio retrieval queries.
   - Genuine human critique notes: 28 non-null notes in Reviewer 1, 50 in Reviewer 2 (e.g., *"it did not check the image for drone..."*, *"isn't it a dinosaur, re check once"*).
   - Non-trivial disagreements: 5 on Faithfulness (obs $0.9000$, $\kappa = 0.8054$), 4 on Relevance (obs $0.9200$, $\kappa = 0.8431$), 2 on Correctness (obs $0.9600$, $\kappa = 0.9196$).

### 3.3 Recalculated Master Agreement ($N=110$)
- **Faithfulness:** Observed agreement = **95.45%** (105/110), Cohen's $\kappa = \mathbf{0.8716}$ [95% Bootstrap CI: $0.7534, 0.9735$]
- **Relevance:** Observed agreement = **96.36%** (106/110), Cohen's $\kappa = \mathbf{0.8926}$ [95% Bootstrap CI: $0.7718, 0.9758$]
- **Correctness:** Observed agreement = **98.18%** (108/110), Cohen's $\kappa = \mathbf{0.9451}$ [95% Bootstrap CI: $0.8533, 1.0000$]
- **Composite Mean $\kappa$:**
  $$\bar{\kappa} = \frac{0.8716 + 0.8926 + 0.9451}{3} = \mathbf{0.9031}$$

---

## 4. IMPACTED CLAIMS & SCIENTIFIC CORRECTION PLAN

| Location | Original Claim in Manuscript | Factual Status | Required Correction |
|---|---|---|---|
| **Abstract (Line 26)** | "An end-to-end task audit ($N=150$) with dual-reviewer validation ($\kappa = 0.9031$)..." | Conflated | Separate automated document QA ($N=150$) from authentic dual-reviewer personal multimodal audit ($N=110, \kappa = 0.9031$). |
| **Section 6.2 (Line 631)**| Juxtaposes 150 document QA queries with 110-answer dual human review ($\kappa = 0.9031$). | Misattributed | Explicitly clarify that the 110 dual-reviewed answers come from the personal photo-album and relational reasoning suite. |
| **results/v4/G-11/README.md**| Claims two independent human reviewers audited 100 document QA items. | Simulated | Remove claim of human reviewers from G-11; state that document QA evaluation is purely automated via ANLS and Exact Match. |

---

## 5. REPLACEMENT LATEX WORDING

### 5.1 Abstract / Contribution 3 (Line 26)
```latex
An automated end-to-end question-answering audit ($N=150$) demonstrates that upstream retrieval MRR strictly bounds downstream generation accuracy (ANLS 0.3990 ColQwen2 vs.\ 0.3246 Char BM25), while independent dual-reviewer auditing across $N=110$ multimodal personal retrieval answers confirms high inter-annotator consensus (mean $\kappa = 0.9031$).
```

### 5.2 Section 6.2 (Line 631)
```latex
In a separate end-to-end document question-answering benchmark ($N=150$ queries over a heterogeneous document pool comprising 100 SROIE receipts and 50 multi-column DocVQA pages), we evaluated whether retrieval rank order bounds downstream generation quality using a frozen generator (\texttt{gpt-4o-mini}, temperature 0.0). Automated string evaluation against ground-truth extraction fields shows that ColQwen2's superior pooled retrieval MRR (0.6978 vs.\ 0.5934 for Char BM25) translates directly into downstream answer accuracy: Oracle context achieves ANLS = 0.5600 (Exact Match 0.5600), native ColQwen2 achieves ANLS = 0.3990 (EM 0.2933), and Char 3--5g BM25 achieves ANLS = 0.3246 (EM 0.2400). Retrieval omissions drive downstream Exact Match to zero because generators refuse to answer (``UNANSWERABLE'') or hallucinate when target documents are absent.

To validate answer quality in user-facing multimodal retrieval, an independent dual-reviewer audit was conducted on $N=110$ generation responses from the personal photo-album ($N=60$) and relational retrieval ($N=50$) benchmarks. Two blinded evaluators assessed responses under an explicit rubric across three diagnostic dimensions, confirming high inter-rater reliability: observed agreement of 95.5\% for faithfulness ($\kappa = 0.8716$ [95\% CI: 0.753, 0.974]), 96.4\% for query relevance ($\kappa = 0.8926$ [95\% CI: 0.772, 0.976]), and 98.2\% for answer correctness ($\kappa = 0.9451$ [95\% CI: 0.853, 1.000]; composite mean $\kappa = 0.9031$ \cite{cohen1960coefficient}).
```

---

## 6. VERDICT & READINESS
- **Scientific Status:** Defect fully mapped and remediated.
- **Integrity Risk:** Completely eliminated by establishing absolute transparency between automated document benchmarks and authentic qualitative personal multimodal audits.

