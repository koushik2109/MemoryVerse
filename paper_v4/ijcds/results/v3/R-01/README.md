# R-01 Deliverable: ColQwen2 Pool and Slice Reconciliation

This deliverable reconciles ColQwen2 evaluations across:
1. Query slices: Verbatim Lookup, Light Paraphrase, Answerable Questions (Natural), and Entity-Abstracted Controls.
2. Pool sizes: 813 distractor pool (with MS-COCO caption fallback) vs 626 pure receipt pool (unannotated distractors).
3. Image conditions: Pristine Clean Document Images vs Physically Degraded Optical Blur Images.

### Key Verified Results
- **Answerable Questions MRR:** Reconciled to 0.8595 [0.825, 0.892]. Char BM25 (0.9471) significantly outperforms ColQwen2 (+0.0876, p = 0.00012). The value 0.0870 belongs strictly to entity-abstracted controls.
- **Light Paraphrase MRR:** 0.4962 [0.448, 0.547] on 813 pool (lift +0.0628 over Word BM25 0.4334, p = 0.0072). On 626 pure receipt pool, ColQwen2 reaches 0.5474 (lift +0.1471 over Word BM25 0.4003, p = 3e-5). The text's +0.0960 mixed the 813 ColQwen2 with the 626 Word BM25.
