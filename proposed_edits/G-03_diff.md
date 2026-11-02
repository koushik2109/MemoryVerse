# Proposed Manuscript Revisions for Gap G3 (Conversational Queries)

These edits directly address reviewer questions regarding Limitation 4 and Table VIII in `paper_v3/ijcds_body.tex`.

---

## 1. Revision for Limitation 4 (`paper_v3/ijcds_body.tex`, Section 6)

### Existing Text:
```latex
\item \textbf{Answerable Question Formulation:} Our answerable queries evaluate natural factual queries isolating semantic abstraction from lexical absence; conversational user queries may exhibit broader diversity.
```

### Proposed Revision:
```latex
\item \textbf{Conversational Query Diversity and Filler Dilution:} While concise human queries preserve high lexical accuracy ($N=200$, Table~\ref{tab:generator2_robustness}), benchmarking across 1,000 queries spanning 4 conversational styles reveals a striking architectural divergence: under multi-sentence conversational dialogue, Word BM25 drops by $-0.1850$ MRR (0.5632 to 0.3782, $-32.8\%$, $p < 0.0001$) and Dense BGE drops by $-0.0594$ (0.4907 to 0.4313) due to conversational filler dilution. Conversely, Character 3--5gram BM25 is immune to conversational padding (retaining 0.5545 vs.\ 0.5689, $97.5\%$ retention), expanding its advantage over Word BM25 to $+0.1763$ MRR ($p < 0.0001$). On underspecified category-only recall, both lexical models drop below 0.13 MRR, where Tuned Hybrid (0.1650) provides the only viable retrieval bridge.
```

---

## 2. Revision for Section 5.B and Table VIII (`paper_v3/ijcds_body.tex`)

### Proposed Addition to Section 5.B:
```latex
\textbf{Conversational Dialogue and Filler Dilution Dynamics:} Extending evaluation beyond concise factoid queries to multi-sentence conversational dialogue ($N=200$ held-out documents, 1,000 queries) uncovers the \emph{conversational dilution deficit}. As conversational padding ("Hey MemoryVerse, could you find my receipt...") enters the query stream, Word BM25 experiences acute term-frequency dilution, dropping from 0.5632 to 0.3782 ($p < 0.0001$). Dense bi-encoders suffer a parallel drop (0.4907 to 0.4313) because preamble tokens induce vector drift away from tabular OCR snippets. In sharp contrast, Character 3--5gram BM25 exhibits structural invariance to conversational filler (0.5689 to 0.5545, diff $-0.0144$, $p > 0.05$), maintaining $+0.1763$ MRR over Word BM25 and $+0.1232$ over Dense BGE. Sub-word character indexing isolates key alphanumeric anchors even when buried inside discursive human prose.
```
