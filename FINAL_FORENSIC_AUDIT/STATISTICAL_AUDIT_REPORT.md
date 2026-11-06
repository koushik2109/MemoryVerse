# Forensic Statistical Audit Report: Holm-Bonferroni and TOST Verification

**Project:** MemoryVerse (IJCDS Submission Verification)  
**Deliverable ID:** `FINAL_FORENSIC_AUDIT/STATISTICAL_AUDIT_REPORT.md`  
**Date:** November 2026  
**Auditor:** Independent Scientific Forensic Auditor  
**Audit Status:** FULL RECALCULATION COMPLETED — DEFECTS CORRECTED  

---

## 1. Executive Summary

This report delivers a complete, independent forensic audit of all 35 statistical hypothesis tests across 6 pre-specified families in the MemoryVerse project, alongside formal Two One-Sided Tests (TOST) for statistical equivalence.

### Key Discoveries & Corrections
1. **Family 4 (MiniLM Reranker) Sorting Bug Corrected:** In previous project files (`results/v4/R-02/holm_family_complete_table.csv`), hypotheses were not sorted in strictly ascending order by $p$-value ($p=0.00010$ was erroneously listed at Rank 7 after $p=1.0$). Properly sorting Family 4 places MiniLM ($p_{\text{raw}} = 0.0090$) at **Rank 3** ($m=7$). At Rank 3, the critical threshold is $\alpha_3 = 0.05 / 5 = 0.0100$. Since $p_{\text{raw}} \le 0.0100$, MiniLM is **statistically significant** ($p_{\text{Holm}} = 0.0450 \le 0.05$), overturning the earlier false claim of non-significance ($p=0.054$).
2. **Family 6 (LLM Bridging) Multiplier Bug Corrected:** In Family 6 ($m=5$), LLM Bridge vs. Sparse BM25 ($N=50$) has $p_{\text{raw}} = 0.0215$. The original draft erroneously multiplied $p_{\text{raw}}$ by 2 to claim $p_{\text{Holm}} = 0.0430$ ("Significant"). In a family of $m=5$, the Rank 1 multiplier is 5, and the critical threshold is $\alpha_1 = 0.05 / 5 = 0.0100$. Because $p_{\text{raw}} = 0.0215 > 0.0100$, this test **fails to reject the null hypothesis** ($p_{\text{Holm}} = 0.1075 > 0.05$). The LLM bridge lift on $N=50$ is **non-significant after family-wise correction**.
3. **Clarification of TOST Equivalence Tests:**
   - **SigLIP vs. 4-Reference Gold on MS-COCO ($N=1,000$):** Evaluated via parametric Student's $t$ TOST with margin $\Delta = \pm 0.05$. Under analytical $\text{SE} = 0.0140$, $t_1 = 2.5857, p_{\text{TOST}} = 0.00493 < 0.01$. Under sample $\text{SE} = 0.01458$, $t_1 = 2.4829, p_{\text{TOST}} = 0.00660 < 0.01$. Both formulations reject non-equivalence at $\alpha = 0.01$. The text has been corrected to cite a **parametric Student's $t$ TOST**, as no permutation TOST was implemented.
   - **Extractive PRF vs. Generative LLM on Multi-Hop ($N=100$):** Under pre-specified margin $\Delta = \pm 0.05$, $t_1 = 2.5000, p_{\text{TOST}} = 0.00703 < 0.01$. The initial $N=50$ subset is underpowered ($p_{\text{TOST}} = 0.1109$), so equivalence is claimed only for the pooled $N=100$ benchmark.

---

## 2. Complete Family-Wise Holm-Bonferroni Audit Ledger

All 35 hypotheses were evaluated under two-tailed paired permutation tests ($N_{\text{perm}} = 100,000$, empirical resolution limit $p = 10^{-5}$). Sequential step-down Holm-Bonferroni correction applies critical threshold $\alpha_k = \frac{\alpha}{m - k + 1}$ and adjusted $p$-value $p_{\text{Holm}} = \min\left(1.0, \max_{j \le k}(p_{\text{raw}, j} \cdot (m - j + 1))\right)$.

The complete corrected table is archived in `FINAL_FORENSIC_AUDIT/STATISTICAL_RECALCULATION.csv`. Below is the family-by-family audit summary:

### Family 1: Sparse OCR Tokenization Comparisons ($m=6, \alpha=0.05$)
All 6 comparisons test Character 3–5gram BM25 vs. Word BM25 across OCR conditions on SROIE ($N=626$).
- Rank 1–6: $p_{\text{raw}} = 0.00001$ across Synthetic 20%, Apple Clean, Degraded Blur, Tesseract 5.5, EasyOCR Clean, and Clean GT.
- All six comparisons: $p_{\text{Holm}} \le 0.00006 < 0.001$.
- **Verdict:** All 6 comparisons **Highly Statistically Significant** ($p < 0.001$).

### Family 2: ColQwen2 vs. Text Proxies ($m=5, \alpha=0.05$)
- Rank 1: ColQwen2 vs. Word BM25 Verbatim ($p_{\text{raw}} = 0.00001, \alpha_1 = 0.0100 \implies p_{\text{Holm}} = 0.00005$, ColQwen2 wins).
- Rank 2: ColQwen2 vs. Tuned Hybrid Verbatim ($p_{\text{raw}} = 0.00001, \alpha_2 = 0.0125 \implies p_{\text{Holm}} = 0.00004$, ColQwen2 wins).
- Rank 3: Char BM25 vs. ColQwen2 Real-OCR Answerable ($p_{\text{raw}} = 0.00110, \alpha_3 = 0.0167 \implies p_{\text{Holm}} = 0.00330$, **Char BM25 wins**).
- Rank 4: ColQwen2 vs. Word BM25 Paraphrase ($p_{\text{raw}} = 0.00720, \alpha_4 = 0.0250 \implies p_{\text{Holm}} = 0.01440$, ColQwen2 wins).
- Rank 5: ColQwen2 vs. Tuned Hybrid Paraphrase ($p_{\text{raw}} = 0.00950, \alpha_5 = 0.0500 \implies p_{\text{Holm}} = 0.00950$, ColQwen2 wins).
- **Verdict:** All 5 comparisons **Statistically Significant**.

### Family 3: MS-COCO Vision vs. Machine Caption Proxies ($m=8, \alpha=0.05$)
- Rank 1–6: SigLIP/CLIP vs. BLIP, BLIP-2, Florence-2 ($p_{\text{raw}} = 0.00001 \implies p_{\text{Holm}} \le 0.00008$, Native Vision wins).
- Rank 7: CLIP vs. Florence-2 Detailed MRR ($p_{\text{raw}} = 0.0380, \alpha_7 = 0.0250 \implies p_{\text{Holm}} = 0.0760 > 0.05$, **Non-Significant**).
- Rank 8: SigLIP vs. 4-Reference Gold Multi-Caption ($p_{\text{raw}} = 0.3210, \alpha_8 = 0.0500$, Null hypothesis retained; evaluated for equivalence via TOST).
- **Verdict:** 6 Significant, 1 Non-Significant, 1 Equivalent.

### Family 4: RAMP Architectural Ablations ($m=7, \alpha=0.05$) [CORRECTED]
- **Rank 1:** Char BM25 + Cross-Encoder Degradation ($p_{\text{raw}} = 0.00010, \alpha_1 = 0.00714 \implies p_{\text{Holm}} = 0.00070$, **Significant Degradation**).
- **Rank 2:** BGE Reranker Lift over Word BM25 ($p_{\text{raw}} = 0.00110, \alpha_2 = 0.00833 \implies p_{\text{Holm}} = 0.00660$, **Statistically Significant**).
- **Rank 3:** MiniLM Reranker Lift over Word BM25 ($p_{\text{raw}} = 0.00900, \alpha_3 = 0.01000 \implies p_{\text{Holm}} = 0.04500$, **Statistically Significant**).
- **Rank 4:** H1 Reliability Weighting on Real Degraded ($p_{\text{raw}} = 0.01890, \alpha_4 = 0.01250 \implies p_{\text{Holm}} = 0.07560$, **Non-Significant; H1 Rejected**).
- **Rank 5:** H2 Surface Router on 20% Syn Noise ($p_{\text{raw}} = 0.04200, \alpha_5 = 0.01667 \implies p_{\text{Holm}} = 0.12600$, **Non-Significant; H2 Rejected**).
- **Rank 6:** H2 Router on Real Degraded Blur ($p_{\text{raw}} = 0.70220, \alpha_6 = 0.02500 \implies p_{\text{Holm}} = 1.00000$, **Zero Lift; H2 Rejected**).
- **Rank 7:** H2 Router on Apple Clean OCR ($p_{\text{raw}} = 1.00000, \alpha_7 = 0.05000 \implies p_{\text{Holm}} = 1.00000$, **Zero Lift; H2 Rejected**).

### Family 5: Cross-Domain FUNSD Form Transfer ($m=4, \alpha=0.05$)
- Rank 1: Char vs. Word BM25 on Apple Clean Paraphrase ($p_{\text{raw}} = 0.0120, \alpha_1 = 0.0125 \implies p_{\text{Holm}} = 0.0480 \le 0.05$, **Significant**).
- Rank 2: Dense BGE vs. Word BM25 on EasyOCR Paraphrase ($p_{\text{raw}} = 0.0180, \alpha_2 = 0.0167 \implies p_{\text{Holm}} = 0.0540 > 0.05$, **Borderline Non-Significant**).
- Rank 3: Dense BGE vs. Word BM25 on Apple Clean Paraphrase ($p_{\text{raw}} = 0.0840, \alpha_3 = 0.0250 \implies p_{\text{Holm}} = 0.1680 > 0.05$, **Non-Significant**).
- Rank 4: Char vs. Word BM25 on Apple Clean Verbatim ($p_{\text{raw}} = 0.3800, \alpha_4 = 0.0500 \implies p_{\text{Holm}} = 0.3800 > 0.05$, **Statistically Tied**).

### Family 6: Multi-Hop Relational Reasoning ($m=5, \alpha=0.05$) [CORRECTED]
- **Rank 1:** LLM Bridge vs. Sparse BM25 ($N=50$) ($p_{\text{raw}} = 0.0215, \alpha_1 = 0.0100 \implies p_{\text{Holm}} = 0.1075 > 0.05$, **NON-SIGNIFICANT**).
- Rank 2: LLM Bridge vs. Single Hybrid ($N=50$) ($p_{\text{raw}} = 1.0000, \alpha_2 = 0.0125 \implies p_{\text{Holm}} = 1.0000$, Statistically Tied).
- Rank 3: LLM Bridge vs. Single Dense ($N=50$) ($p_{\text{raw}} = 1.0000, \alpha_3 = 0.0167 \implies p_{\text{Holm}} = 1.0000$, Identical hits 32/50).
- Rank 4: Extractive PRF vs. LLM Bridge ($N=50$) ($p_{\text{raw}} = 1.0000$, Underpowered for equivalence; $p_{\text{TOST}} = 0.1109$).
- Rank 5: Extractive PRF vs. LLM Bridge ($N=100$ Pooled) ($p_{\text{raw}} = 1.0000, p_{\text{TOST}} = 0.00703 < 0.01$, **TOST Equivalent**).

---

## 3. Two One-Sided Tests (TOST) Equivalence Proofs

### Proof 1: SigLIP vs. 4-Reference Gold Caption on MS-COCO ($N=1,000$)
- **Null Hypothesis ($H_0$):** $|\mu_{\text{SigLIP}} - \mu_{\text{Gold}}| \ge \Delta$, where $\Delta = 0.0500$ MRR.
- **Alternative Hypothesis ($H_1$):** $-\Delta < \mu_{\text{SigLIP}} - \mu_{\text{Gold}} < \Delta$.
- **Observed Values:** $\hat{\mu}_{\text{SigLIP}} = 0.6071$, $\hat{\mu}_{\text{Gold}} = 0.5933$, observed diff $\hat{D} = +0.0138$.
- **Standard Error:** Analytical $\text{SE} = 0.0140$; Empirical sample $\text{SE} = 0.01458$.
- **Test Statistics:**
  $$t_1 = \frac{\Delta - |\hat{D}|}{\text{SE}} = \frac{0.0500 - 0.0138}{0.0140} = 2.5857$$
  $$p_{\text{TOST}} = 1 - \Phi(t_1, \text{df}=999) = 0.00493 < 0.01$$
- Using sample SE ($0.01458$): $t_1 = 2.4829, p_{\text{TOST}} = 0.00660 < 0.01$.
- **Conclusion:** Both formulations reject non-equivalence at $\alpha = 0.01$. SigLIP visual retrieval and 4-Reference human descriptions are statistically equivalent within $\pm 0.05$ MRR.

### Proof 2: Extractive PRF vs. Generative LLM on Multi-Hop ($N=100$)
- **Null Hypothesis ($H_0$):** $|\text{Hit@10}_{\text{PRF}} - \text{Hit@10}_{\text{LLM}}| \ge \Delta$, with $\Delta = 0.0500$.
- **Observed Values:** $\text{Hit@10}_{\text{PRF}} = 0.8200$, $\text{Hit@10}_{\text{LLM}} = 0.8200$, $\hat{D} = 0.0000$.
- **Standard Error:** Paired difference variance yields $\text{SE} = 0.0200$.
- **Test Statistic:**
  $$t_1 = \frac{0.0500 - 0.0000}{0.0200} = 2.5000, \quad p_{\text{TOST}} = 1 - t_{99}(2.5000) = 0.00703 < 0.01$$
- **Conclusion:** Rejects non-equivalence at $\alpha = 0.01$. Extractive PRF achieves equivalent retrieval coverage to LLM query expansion while running $394\times$ faster.

---

## 4. Remediation Checklist for Manuscript

1. **Update MiniLM in Section IV-C and Table III:** Change description from "Borderline Non-Significant ($p=0.054$)" to "Statistically Significant ($p_{\text{Holm}} = 0.0450 \le 0.05$)".
2. **Update LLM Bridge in Section IV-D and Table IV:** State explicitly that while LLM Bridging yields $+8$ net hits over Sparse BM25 on the initial $N=50$ slice ($p_{\text{raw}} = 0.0215$), this difference does **not** survive family-wise correction ($p_{\text{Holm}} = 0.1075 > 0.05, m=5$).
3. **Clarify TOST Methodology:** Ensure all textual references describe TOST as a **parametric Student's $t$ test**, preventing reviewer skepticism regarding undocumented permutation test harnesses.

