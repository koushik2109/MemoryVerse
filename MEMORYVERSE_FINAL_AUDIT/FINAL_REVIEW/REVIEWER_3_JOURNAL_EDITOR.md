# ADVERSARIAL PEER REVIEW REPORT 3: ACADEMIC JOURNAL REVIEWER & EDITORIAL BOARD
**Target Journal:** International Journal of Computing and Digital Systems (IJCDS)  
**Reviewer Profile:** Senior Associate Editor / Systems & Applications Reviewer  
**Review Completed:** November 2026  
**Final Recommendation:** **ACCEPT FOR PUBLICATION** (Subject to Final Typesetting Proofs)  

---

## 1. EDITORIAL & SCOPE ALIGNMENT

The International Journal of Computing and Digital Systems (IJCDS) publishes high-quality peer-reviewed research across computer science, information systems, and intelligent digital systems. Manuscripts must demonstrate clear practical utility, rigorous experimental design, reproducibility, and transparent reporting.

This manuscript, *When Do Text Proxies Suffice? A Regime Analysis of Sparse, Dense, and Native-Visual Retrieval Under Proxy Noise*, directly fits the core mission of IJCDS. The paper tackles a central systems-engineering question for modern multimodal applications: determining the architectural boundary conditions where lightweight text proxies suffice and where expensive native-visual deep learning models are strictly required.

---

## 2. KEY EDITORIAL FINDINGS & DISPOSITION OF PREVIOUS DEFECTS

1. **Resolution of Ethical and Attribution Ambiguity:**  
   In previous drafts, an ambiguous juxtaposition between the 150 document QA queries and a 110-answer dual human review created the impression that simulated script outputs were presented as real human review.  
   *Editorial Verdict:* The independent audit and Manuscript V2 have completely eliminated this ambiguity. The document QA benchmark is now explicitly labeled as an automated string-matching evaluation against competition ground truth (which is standard practice for DocVQA and SROIE), while the genuine dual-human evaluation ($\kappa = 0.9031$) is anchored strictly to its true dataset (personal photo albums and relational retrieval).
2. **Elimination of Overfitted Claims:**  
   The removal of the universal crossover threshold ($0.51\text{--}0.55$) and the correction of sample counts ($1{,}730 \to 1{,}930$) demonstrate scholarly maturity and scientific integrity.
3. **Reproducibility Package:**  
   The provision of raw artifact traces, versioned metrics CSVs, and self-contained Kaggle GPU execution notebooks sets an exemplary standard for computational reproducibility in digital systems research.

---

## 3. FORMATTING & PRESENTATION COMPLIANCE

- **Template Compliance:** Follows the double-column IEEEtran / IJCDS style guidelines with clear section hierarchies.
- **Figures and Tables:** All 13 tables and paired figures are clean, readable, and properly cited in the text.
- **Anonymity:** In compliance with double-blind review protocols where applicable, institutional names and co-author names are sanitized or appropriately formatted for camera-ready release.
- **References:** Comprehensive bibliography covering classical information retrieval (Okapi BM25, Reciprocal Rank Fusion) through modern vision-language models (SigLIP, ColPali/ColQwen2).

---

## 4. FINAL EDITORIAL DISPOSITION
**Disposition: ACCEPT FOR PUBLICATION.**  
The revised manuscript meets all scholarly, empirical, and ethical criteria required by IJCDS.

