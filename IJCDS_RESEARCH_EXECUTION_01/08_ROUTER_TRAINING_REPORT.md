# ADAPTIVE ROUTER AUDIT & NEGATIVE ABLATION REPORT
- Investigated learned routing based on OCR confidence, query length, and token overlap (Table IV, H4.5-H4.7).
- Empirical test results on held-out data showed non-significant gains (pHolm = 0.1260 to 0.8200).
- Under HI-03 (Approved Option A), the negative router finding is retained as an empirical ablation.
- The project advocates the fixed two-stage cascade (K1=50), which achieves 15.5x measured speedup on CPU (12.18 ms) and retains 97.2% visual MRR without classifier overhead.
