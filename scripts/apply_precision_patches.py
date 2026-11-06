import os, shutil

# Copy clean base from ijcds/ijcds_body.tex to FINAL_IJCDS_MANUSCRIPT_SOURCE/ijcds_body.tex
shutil.copyfile("ijcds/ijcds_body.tex", "FINAL_IJCDS_MANUSCRIPT_SOURCE/ijcds_body.tex")

with open("FINAL_IJCDS_MANUSCRIPT_SOURCE/ijcds_body.tex", "r", encoding="utf-8") as f:
    c = f.read()

# 1. Page 4 line trimming to balance the bootstrap addition
t_chunk = "For \\texttt{bge-base-en-v1.5} and \\texttt{all-MiniLM-L6-v2}, maximum sequence length is set to 256; on SROIE receipts, while 33.39\\% of receipts exceed 256 WordPiece tokens, chunked MaxP evaluation across 150-word sliding windows ($\\le 256$ tokens) yields $<0.007$ MRR difference from unchunked single-vector retrieval (0.2030 chunked vs.\\ 0.2056 unchunked on MiniLM; 0.3028 vs.\\ 0.3022 on BGE-Base; Table~\\ref{tab:dense_formulation_v6}), confirming that sequence truncation does not explain dense retrieval deficits."
r_chunk = "For \\texttt{bge-base-en-v1.5} and \\texttt{all-MiniLM-L6-v2}, maximum sequence length is set to 256; chunked MaxP evaluation across 150-word sliding windows yields $<0.007$ MRR difference from unchunked single vectors (0.2030 vs.\\ 0.2056 on MiniLM; 0.3028 vs.\\ 0.3022 on BGE; Table~\\ref{tab:dense_formulation_v6}), confirming that sequence truncation does not explain dense retrieval deficits."
assert t_chunk in c, "t_chunk not found"
c = c.replace(t_chunk, r_chunk, 1)

# 2. Bootstrap description distinguishing query-level from cluster-level
t_boot = "Confidence intervals (95\\% CI) are computed via cluster bootstrap resampling at the document level (10,000 iterations). Equivalence is evaluated using Two One-Sided Tests (TOST) with pre-specified margins."
r_boot = "Confidence intervals (95\\% CI) use 10,000 bootstrap iterations: query-level for single-probe benchmarks (MS-COCO, DocVQA, FUNSD) and document-cluster bootstrap for multi-probe pools (SROIE, conformal partitions) to prevent variance underestimation from intra-document correlation. Equivalence is evaluated using Two One-Sided Tests (TOST) with pre-specified margins."
assert t_boot in c, "t_boot not found"
c = c.replace(t_boot, r_boot, 1)

# 3. Table captions for bootstrap clarity
c = c.replace("95\\% Cluster Bootstrap CIs", "95\\% Document-Cluster Bootstrap CIs")
c = c.replace("Cross-modal text-to-image retrieval on $N=1{,}000$ MS-COCO Karpathy-test images.", "Cross-modal text-to-image retrieval on $N=1{,}000$ MS-COCO Karpathy-test images (95\\% Query-Level Bootstrap CIs).")
c = c.replace("$N=500$ Document Pages, 95\\% Bootstrap CIs; measured", "$N=500$ Document Pages, 95\\% Query-Level Bootstrap CIs; measured")

# 4. Table XII DocVQA ColQwen2 metric
t_tab12 = "Document Visual Form & Visual Layout & Visual Question Answering & \\textbf{ColQwen2-v1.0 (GPU)} & 0.6450 [0.601, 0.689] & ColQwen2 standalone & On DocVQA, native visual retrieval substantially outperforms text proxies ($\\le 0.3507$) where layout geometry governs answerability. \\\\"
r_tab12 = "Document Visual Form & Visual Layout & Visual Question Answering & \\textbf{ColQwen2-v1.0 (GPU)} & 0.5497 [0.511, 0.589] & ColQwen2 standalone & On DocVQA, native visual retrieval substantially outperforms text proxies ($\\le 0.3507$) where layout geometry governs answerability. \\\\"
assert t_tab12 in c, "t_tab12 not found"
c = c.replace(t_tab12, r_tab12, 1)

# 5. Line 605 DocVQA ColQwen2 metric
t_l605 = "(DocVQA MRR 0.6450 vs.\\ proxies $\\le 0.3507$, noting in-domain training overlap)"
r_l605 = "(DocVQA MRR 0.5497 vs.\\ proxies $\\le 0.3507$, noting in-domain training overlap)"
assert t_l605 in c, "t_l605 not found"
c = c.replace(t_l605, r_l605, 1)

# 6. Appendix A Family 3 confirmatory tests
t_app3 = "\\textbf{(3) Family 3: Native Vision vs.\\ Captions on MS-COCO ($m=8$):} SigLIP vs.\\ BLIP/BLIP-2/Florence-2 ($p < 10^{-4}$); CLIP vs.\\ BLIP ($p < 10^{-4}$); CLIP vs.\\ Florence-2 (raw $p=0.0380$, $p_{\\text{Holm}}=0.0760$, non-significant after correction); SigLIP vs.\\ 4-Ref Multi-Caption Gold (paired $p=0.321$, $p_{\\text{TOST}}=0.0050$ at $\\Delta=\\pm 0.05$, equivalent)."
r_app3 = "\\textbf{(3) Family 3: Native Vision vs.\\ Captions on MS-COCO ($m=8$):} SigLIP vs.\\ BLIP MiniLM/BGE ($p < 10^{-4}$); CLIP vs.\\ BLIP MiniLM/BGE ($p < 10^{-4}$); SigLIP vs.\\ CLIP ($p < 10^{-4}$); SigLIP vs.\\ Single Gold BGE ($p < 10^{-4}$); SigLIP vs.\\ 4-Ref Gold BGE (diff $+0.0236$, raw $p=0.0905$, $p_{\\text{Holm}}=0.1809$, tied); SigLIP vs.\\ 4-Ref Gold Hybrid (diff $-0.0007$, raw $p=0.9563$, $p_{\\text{Holm}}=0.9563$, tied)."
assert t_app3 in c, "t_app3 not found"
c = c.replace(t_app3, r_app3, 1)

with open("FINAL_IJCDS_MANUSCRIPT_SOURCE/ijcds_body.tex", "w", encoding="utf-8") as f:
    f.write(c)

print("Applied precision patches successfully.")

