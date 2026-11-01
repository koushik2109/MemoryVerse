# Proposed Manuscript Diff: T2-03 Isolate "Sub-Word" vs "Character" Effects

## Target Location
- `ijcds_body.tex` (Section I, Contribution 1, Line 35; Section IV-A, Line 295; Section V-A, Line 510)

## Context & Motivation
Reviewers may ask whether the robustness of character-level BM25 is simply an effect of sub-word tokenization
(such as BERT WordPiece or GPT-2 BPE) or whether contiguous character n-grams are the true causal driver.
Experiment **T2-03** evaluated 11 tokenization and expansion schemes: Char n=2, 3, 4, 5, 6, Char 3–5g,
WordPiece (BERT), BPE (GPT-2), Word + Porter Stemming, Word + Levenshtein Fuzzy Expansion, and Word Standard.

## Proposed LaTeX Fragment Diff
```diff
--- a/ijcds_body.tex (Section IV-A, Line 295)
+++ b/ijcds_body.tex (Section IV-A, Line 295)
+\subsection{Ablation of Indexing Granularity: Disentangling Character $n$-grams from Subwords}
+\label{sec:subword_ablation}
+To test whether the noise tolerance of character BM25 is an artifact of sub-word tokenization or a distinct property of dense overlapping substrings, we benchmarked 11 token representations across optical blur and synthetic noise (Table~\ref{tab:subword_ablation}):
+\begin{enumerate}
+    \item \textbf{Failure of Learned Subwords under Noise:} Learned subword tokenizers (BERT WordPiece and GPT-2 BPE) fail to replicate character $n$-gram resilience. Under optical blur, WordPiece drops to $0.6608$ MRR and BPE drops to $0.5612$ MRR, lagging Char 3--5g BM25 ($0.8086$) by over $-0.14$ to $-0.24$ MRR. Under 20\% synthetic OCR noise, WordPiece ($0.4095$) and BPE ($0.3930$) collapse toward Word BM25 ($0.3155$), while Char 3--5g retains $0.6723$ MRR. Learned subwords rely on precise vocabulary split rules; a single corrupted character causes catastrophic out-of-vocabulary fallback to isolated characters with skewed inverse document frequencies.
+    \item \textbf{Optimal Character $n$-gram Order:} Individual character $n$-grams demonstrate that $n=3$ ($0.7972$ blur MRR) and $n=4$ ($0.8070$ blur MRR) provide the primary noise-shielding capacity. Bigrams ($n=2$) suffer severe false-positive collisions ($0.6136$ blur MRR), whereas $n \ge 5$ undergoes error fragmentation ($0.7759$ for $n=6$). The combined window $n \in [3, 5]$ achieves optimal performance across both clean ($0.9437$) and degraded ($0.8086$) regimes.
+    \item \textbf{Linguistic Expansions:} Neither Porter stemming ($0.6824$ blur MRR) nor Levenshtein fuzzy expansion ($0.6803$ blur MRR) closes the gap with character $n$-grams. Receipt entities (store codes, amounts, dates) are non-inflected proper nouns, rendering stemming ineffective, while fuzzy expansion increases distractor collision rates.
+\end{enumerate}
+We conclude that \emph{character $n$-gram} is the scientifically accurate term: learned subwords do not confer OCR proxy resilience.
```
