with open('.agents/memory.md', 'r', encoding='utf-8') as f:
    text = f.read()

pos = text.find('### 2026-10-29: Final IJCDS Submission Audit')
if pos != -1:
    text = text[:pos]

clean_entry = """### 2026-10-29: Final IJCDS Submission Audit, Numerical Reconciliation, and Page 15 Bibliography Repair
- **Root Cause of Page 15 Bibliography Distortion:** `widetext.sty` in `ijcds_header.tex` loaded `flushend.sty`, which attempted destructive last-page column balancing, splitting the 46 references into interleaved blocks ([1]-[9] and [15]-[29] in col 1; [10]-[14] and [30]-[46] in col 2) and generating four `[leftcolumn badness]: 1000000` errors. Removing `widetext` restored standard IEEEtran two-column flow ([1]-[29] in Col 1, [30]-[46] in Col 2) with pristine legibility and zero overlaps.
- **Table XV & Right-Column Collision Resolved:** Table XV (Preliminary Feasibility Benchmarks) had a 75pt overfull hbox protruding into Column 2, colliding with Section 7 text. Resized cell widths using `\\makecell`, tightened column headers, and reduced font to `\\fontsize{6.5pt}{7.5pt}\\selectfont`, completely eliminating the overfull hbox and locking Table XV inside `\\columnwidth`.
- **Figure 3 & Figure 4 Isolation:** Moved Figure 4 float declaration into Section V-B to decouple it from Figure 3. Figure 3 is cleanly placed on Page 7, and Figure 4 is on Page 8.
- **Section 8 & Page 14 Clean Finish:** Tightened Section 7 enumeration and Section 8 conclusion text, pulling all narrative onto Page 14 so that Page 15 starts cleanly at the top with `References`. Total page count remains strictly 15 pages (10-15 page IJCDS limit).
- **Numerical Reconciliation:**
  - Table II vs Narrative: Clarified that on answerable questions with intact topical anchors, CPU lexical retrieval (Char BM25 0.9471, Tuned Hybrid 0.9125) outperforms ColQwen2 (0.8595, paired diff -0.0855, p=0.00012, Table II). The diff +0.0051 (p_Holm=1.0000) belongs strictly to the secondary entity-abstracted control.
  - FUNSD Table X vs Prose: Documented that the 600-distractor pool gives Apple Char BM25 = 0.4892 (Tesseract 0.4725) and Word BM25 = 0.3643 (Tesseract 0.3412), whereas the 0.8433/0.8066/0.8364 values represent the closed 100-form pool without distractors.
  - Table IV Completeness: Added the two missing ablation rows (`+ Learned Router (H2)` = 0.3765* and `+ Heuristic Reliability Weighting` = 0.3574 syn / 0.6368* degraded blur).
- **Deliverable Synchronization:** Successfully recompiled `main.pdf` (15 pages) and updated `ijcds/ijcds_master_doc.pdf` and `~/Downloads/MemoryVerse_IJCDS_Submission.pdf`.
"""

with open('.agents/memory.md', 'w', encoding='utf-8') as f:
    f.write(text.rstrip() + '\n\n' + clean_entry)

print("Memory updated cleanly from script!")

