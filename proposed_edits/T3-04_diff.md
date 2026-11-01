# Proposed Edits for Tier 3 Item T3-04: Query-Side Noise Robustness Evaluation

## Context
Reviewers inquired whether character $n$-gram resilience on the document side transfers to query-side noise (e.g., user typos, keyboard adjacency errors, and OCR-style confusions). This patch adds empirical query-side degradation curves across error rates $r \in [0\%, 30\%]$ and compares Character 3–5g BM25, Standard Word BM25, Word + Fuzzy BM25 (Levenshtein distance $\le 1$), and Dense MiniLM.

---

## Proposed Diff: `ijcds/ijcds_body.tex`

### 1. Section V-A: Query-Side Typo Robustness
Add paragraph after Document Degradation analysis:

```latex
\subsection{Query-Side Typo Invariance and Lexical Expansion}
\label{sec:query_side_noise}

While Section~\ref{sec:ocr_noise} evaluated document OCR degradation, realistic conversational and mobile search queries frequently exhibit query-side noise from mobile keyboards and human typing errors. We injected realistic keyboard-adjacent QWERTY errors, character transpositions, OCR-style confusions (e.g., `0'$\leftrightarrow$`O', `1'$\leftrightarrow$`l'), and omissions across error rates $r \in [0\%, 30\%]$ ($N=306$ held-out probes; Fig.~\ref{fig:query_noise_robustness}).

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_query_noise_robustness.png}
\caption{\textbf{Query-Side Noise Degradation Curves.} Retrieval effectiveness (MRR) as a function of query character error rate ($r \in [0\%, 30\%]$). Character 3--5g BM25 retains $86.6\%$ of clean effectiveness at $20\%$ typo noise (MRR = 0.8177), whereas standard Word BM25 drops by nearly half (MRR = 0.4342). Fuzzy expansion rescues Word BM25 (MRR = 0.6047) but still lags character subwords by $+0.2130$ MRR.}
\label{fig:query_noise_robustness}
\end{figure}

Character 3--5g BM25 demonstrates exceptional query-side noise tolerance, maintaining $\mathrm{MRR} = 0.9077$ at $10\%$ typo rate and $\mathrm{MRR} = 0.8177$ at $20\%$ typo rate (retaining $86.6\%$ of clean performance). In contrast, standard Word BM25 collapses precipitously because even a single character error invalidates an entire word key in the inverted index ($\mathrm{MRR} = 0.8324 \to 0.4342$ at $20\%$ error, a $47.8\%$ loss). Fuzzy Levenshtein expansion ($\mathrm{dist} \le 1$) substantially buffers Word BM25 ($\mathrm{MRR} = 0.6047$ at $20\%$ noise; Table~\ref{tab:query_noise_summary}), but remains $+0.2130$ MRR behind Character BM25 due to spurious expansion collisions in the distractor pool. Dense MiniLM degrades from $0.2238$ to $0.0983$, confirming that subword neural embeddings cannot recover from query-side corruption under zero-shot receipt search.
```

