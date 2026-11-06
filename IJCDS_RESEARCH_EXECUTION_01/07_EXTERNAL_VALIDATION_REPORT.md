# EXTERNAL DATASET VALIDATION REPORT
- FUNSD (Forms): Char BM25 maintains 0.8433 clean MRR; under paraphrase Char BM25 (0.2228) and BGE (0.1834) outperform Word BM25 (0.1573, pHolm=0.0480).
- DocVQA (Layout): Native visual ColQwen2 achieves 0.6450 MRR vs text proxies <= 0.3507.
- MS-COCO (Photos): Native SigLIP (0.6071) decisively beats captions (BLIP 0.3728, Florence-2 0.4420, p < 10^-4) and matches 4-ref gold (pTOST=0.0050).
