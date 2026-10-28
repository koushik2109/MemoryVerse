# Qualitative Failure Taxonomy: Case Studies Across Retrieval Regimes

### Case 1: Sparse Lexical OOV under Heavy Paraphrasing
- **Probe ID:** probe_0004 | **Target ID:** sroie_004
- **Query:** 'Which convenience store in Petaling Jaya supplied items on 12/04/2018?'
- **Ground Truth Text:** '99 SPEED MART S/B ... 12-04-18 ... RM 14.50'
- **Mechanism:** Word BM25 fails because query uses full month name '04' vs document '04-18' and abstract term 'convenience store' absent from invoice. Dense BGE-base resolves target at Rank 1.

### Case 2: Dense Numeric Hallucination / Drift
- **Probe ID:** probe_0012 | **Target ID:** sroie_012
- **Query:** 'Total RM 24.80'
- **Retrieved Top-3:** sroie_088 (RM 24.00), sroie_104 (RM 24.50), sroie_312 (RM 24.80 at Rank 3)
- **Mechanism:** WordPiece subword tokenization merges decimal components, causing dense embedding proximity to drift between nearby numerical amounts. Pure BM25 resolves exact token at Rank 1.

### Case 3: Optical Character Transposition under Blur
- **Probe ID:** probe_0035 | **Target ID:** sroie_035
- **Query:** 'RESTORAN KIN KIN'
- **OCR Text:** 'RESTORAN K1N K1N (optical digit substitution)'
- **Mechanism:** Word BM25 drops rank to 14; Character 3-5gram BM25 retains Rank 1 due to overlapping character trigrams ('kin', 'in ').

