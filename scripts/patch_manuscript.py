import os

path = "FINAL_IJCDS_MANUSCRIPT_SOURCE/ijcds_body.tex"
with open(path, "r", encoding="utf-8") as f:
    c = f.read()

# 1. Adjust vspaces around sections 7, 8, appendices
c = c.replace("\\vspace{3.5mm}\n\\section{Limitations", "\\vspace{1.5mm}\n\\section{Limitations", 1)
c = c.replace("\\vspace{3.5mm}\n\\section{Conclusion", "\\vspace{1.5mm}\n\\section{Conclusion", 1)
c = c.replace("\\vspace{4.0mm}\n\\appendices", "\\vspace{1.5mm}\n\\appendices", 1)

# 2. Tighten conclusion ending
t_conc = "On 100 relational queries, extractive PRF statistically matches generative LLM bridging (Hit@10 0.820 vs.\\ 0.820, $p_{\\text{TOST}}=0.0073 < 0.01$ at $\\Delta=\\pm 0.05$) while eliminating a $394\\times$ latency multiplier. Formalizing these trade-offs provides an engineering roadmap for efficient, trustworthy multimodal retrieval."
r_conc = "On 100 relational queries, extractive PRF matches generative LLM bridging (Hit@10 0.820 vs.\\ 0.820, $p_{\\text{TOST}}=0.0073 < 0.01$ at $\\Delta=\\pm 0.05$) eliminating a $394\\times$ latency multiplier. Formalizing these trade-offs provides a disciplined roadmap for efficient multimodal retrieval."
if t_conc in c:
    c = c.replace(t_conc, r_conc, 1)
    print("Updated conclusion")
else:
    print("t_conc not found")

# 3. Update Appendix A Family 3
t_app3 = "\\textbf{(3) Family 3: Native Vision vs.\\ Captions on MS-COCO ($m=8$):} SigLIP vs.\\ BLIP/BLIP-2/Florence-2 ($p < 10^{-4}$); CLIP vs.\\ BLIP ($p < 10^{-4}$); CLIP vs.\\ Florence-2 (raw $p=0.0380$, $p_{\\text{Holm}}=0.0760$, non-significant after correction); SigLIP vs.\\ 4-Ref Multi-Caption Gold (paired $p=0.321$, $p_{\\text{TOST}}=0.0050$ at $\\Delta=\\pm 0.05$, equivalent)."
r_app3 = "\\textbf{(3) Family 3: Native Vision vs.\\ Captions on MS-COCO ($m=8$):} SigLIP vs.\\ BLIP MiniLM/BGE ($p < 10^{-4}$); CLIP vs.\\ BLIP MiniLM/BGE ($p < 10^{-4}$); SigLIP vs.\\ CLIP ($p < 10^{-4}$); SigLIP vs.\\ Single Gold BGE ($p < 10^{-4}$); SigLIP vs.\\ 4-Ref Gold BGE (diff $+0.0236$, raw $p=0.0905$, $p_{\\text{Holm}}=0.1809$, tied); SigLIP vs.\\ 4-Ref Gold Hybrid (diff $-0.0007$, raw $p=0.9563$, $p_{\\text{Holm}}=0.9563$, tied)."
if t_app3 in c:
    c = c.replace(t_app3, r_app3, 1)
    print("Updated Appendix A Family 3")
else:
    print("t_app3 not found")

# 4. Update Table XII
t_tab12 = "0.6450 [0.601, 0.689] & ColQwen2 standalone & On DocVQA, native visual retrieval substantially outperforms text proxies ($\\le 0.3507$) where layout geometry governs answerability."
r_tab12 = "0.5497 [0.511, 0.589] & ColQwen2 standalone & On DocVQA, native visual retrieval substantially outperforms text proxies ($\\le 0.3507$) where layout geometry governs answerability."
if t_tab12 in c:
    c = c.replace(t_tab12, r_tab12, 1)
    print("Updated Table XII")
else:
    print("t_tab12 not found")

# 5. Update line 605
t_line605 = "(DocVQA MRR 0.6450 vs.\\ proxies $\\le 0.3507$, noting in-domain training overlap)"
r_line605 = "(DocVQA MRR 0.5497 vs.\\ proxies $\\le 0.3507$, noting in-domain training overlap)"
if t_line605 in c:
    c = c.replace(t_line605, r_line605, 1)
    print("Updated line 605")
else:
    print("t_line605 not found")

# 6. Table captions for bootstrap clarity
c = c.replace("95\\% Cluster Bootstrap CIs", "95\\% Document-Cluster Bootstrap CIs")
c = c.replace("Cross-modal text-to-image retrieval on $N=1{,}000$ MS-COCO Karpathy-test images.", "Cross-modal text-to-image retrieval on $N=1{,}000$ MS-COCO Karpathy-test images (95\\% Query-Level Bootstrap CIs).")
c = c.replace("$N=500$ Document Pages, 95\\% Bootstrap CIs; measured", "$N=500$ Document Pages, 95\\% Query-Level Bootstrap CIs; measured")

with open(path, "w", encoding="utf-8") as f:
    f.write(c)
print("Finished patching FINAL_IJCDS_MANUSCRIPT_SOURCE/ijcds_body.tex")

