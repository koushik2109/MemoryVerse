# FINAL QUERY RELEVANCE AUDIT: HUMAN EVALUATION GROUND TRUTH & ADJUDICATION

**Audit Scope:** Human evaluation query suite in `data/human_queries_200.csv` ($N=200$ receipt targets, $N=400$ query instances)  
**Adjudication Artifact:** [`data/human_queries_adjudicated.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/data/human_queries_adjudicated.csv)  
**Metric Summary Table:** [`data/human_queries_relevance_summary.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/data/human_queries_relevance_summary.csv)  
**Lead Auditor:** Senior Information Retrieval Researcher & Statistical Auditor  
**Date:** November 2026  
**Status:** COMPLETED, RECORD-ADJUDICATED, AND REPRODUCED  

---

## 1. Executive Summary & Core Findings

1. **Query Suite Composition (200 Targets / 400 Instances):**
   The suite comprises $N=200$ held-out SROIE receipt probes, each evaluated across two distinct query formulations:
   - `handwritten_verbatim` ($N=200$ instances): normalized vendor name + city/state anchor.
   - `handwritten_paraphrase` ($N=200$ instances): natural user search intent with category context.
   Together, these constitute $N=400$ evaluated query instances across 200 target receipts.

2. **Provenance Partitioning (Tier 1 vs. Tier 2):**
   - **Tier 1: Manual Inspection Probes (Rows 0–99, $N=200$ instances):** Manually authored by human evaluators who inspected receipt images and OCR line items (`results/v7/human_queries_template.csv`). Probes frequently reference itemized purchase details (e.g., probe `sroie_000` Book Ta K: *"modelling clay"*).
   - **Tier 2: Domain-Templated Probes (Rows 100–199, $N=200$ instances):** Programmatically generated in Git commit `44abfbf9` (`scripts/build_human_queries_200.py`) using rule-based templates based on vendor category and Malaysian town metadata.

3. **Duplication & Under-Specification:**
   Across the 400 instances, there are only **290 unique query strings**. Exactly **37 distinct strings** are repeated across rows, representing **147 query instances** (36.8% of the suite).
   - Because the SROIE dataset contains multiple receipts from recurring commercial retail chains (e.g., 23 receipts from Gardenia Bakeries in Shah Alam, 7 receipts from 99 Speedmart in Klang, 7 from MR. D.I.Y. in Seri Kembangan), identical query strings were assigned to distinct receipt IDs.
   - None of the repeated queries contain transaction-unique anchors (such as invoice number, date, or itemized payment amount). They are inherently non-discriminating known-item queries.

4. **False-Negative Risk & Adjudication:**
   Under strict single-target ground truth, retrieving a valid receipt from the queried merchant at Rank 1 is penalized as a retrieval failure if it is not the single arbitrary row-assigned target ID.
   Rather than performing coarse metadata pooling, an independent record-by-record semantic adjudication was executed ([`data/human_queries_adjudicated.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/data/human_queries_adjudicated.csv)). For each duplicate query string, candidate receipts were evaluated against the query's explicit textual constraints.

---

## 2. Inventory of Repeated Query Groups

The table below summarizes the top repeated query groups, the number of instances, and their adjudicated status:

| Repeated Query String | Generation Tier | Instances | Target Receipts | Disambiguating Information Present? | Adjudication Rationale |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `Bakery delivery and bread invoice from Gardenia in Shah Alam` | Tier 2 Paraphrase | 23 | `sroie_328`, `330`, `332`, `333`, `334`, `337`, `342`, `343`, `344`, `348`, `352`, `353`, `362`, `363`, `365`, `366`, `367`, `368`, `369`, `370`, `372`, `373`, `374` | No (No date/invoice) | All 23 receipts are delivery invoices from Gardenia in Shah Alam; all satisfy the semantic request. |
| `Gardenia Bakeries (Kl) Shah Alam` | Tier 2 Verbatim | 22 | `sroie_330`, `332`, `333`, `334`, `337`, `342`, `343`, `344`, `348`, `352`, `353`, `362`, `363`, `365`, `366`, `367`, `368`, `369`, `370`, `372`, `373`, `374` | No | All associated receipts share exact vendor entity and town. |
| `Home improvement tools and hardware supplies receipt from MR DIY in Seri Kembangan` | Tier 2 Paraphrase | 7 | `sroie_231`, `378`, `379`, `389`, `390`, `392`, `401` | No | All 7 receipts are from MR. D.I.Y. in Seri Kembangan. |
| `99 Speed Mart Klang` | Tier 2 Verbatim | 7 | `sroie_259`, `276`, `387`, `391`, `394`, `398`, `405` | No | All 7 receipts are from 99 Speed Mart branches in Klang. |
| `Grocery shopping receipt at 99 Speedmart convenience store in Klang` | Tier 2 Paraphrase | 7 | `sroie_259`, `276`, `387`, `391`, `394`, `398`, `405` | No | All satisfy retail grocery shopping at 99 Speedmart in Klang. |
| `Mr. D.I.Y. (Kuchai) Seri Kembangan` | Tier 2 Verbatim | 5 | `sroie_231`, `378`, `379`, `390`, `401` | No | All 5 receipts are from MR. D.I.Y. Kuchai branch. |
| `Syarikat Perniagaan Gin Kee hardware` | Tier 1 Verbatim | 4 | `sroie_099`, `101`, `114`, `122` | No | All 4 receipts are from Syarikat Perniagaan Gin Kee. |
| `Syarikat Perniagaan Gin Kee Setapak` | Tier 1 Verbatim | 4 | `sroie_111`, `121`, `124`, `127` | No | All 4 receipts are from Gin Kee Setapak store. |
| `Segi Cash & Carry Shah Alam` | Tier 2 Verbatim | 4 | `sroie_239`, `242`, `243`, `264` | No | All 4 receipts are from Segi Cash & Carry in Shah Alam. |
| `Wholesale sundry goods and household groceries purchase at Segi Cash & Carry` | Tier 2 Paraphrase | 4 | `sroie_239`, `242`, `243`, `264` | No | All satisfy wholesale grocery purchase at Segi. |
| `Unihakka International Johor` | Tier 2 Verbatim | 4 | `sroie_283`, `284`, `292`, `295` | No | All 4 receipts are Unihakka dining bills in Johor. |
| `Store shopping and payment receipt from Unihakka International in Johor` | Tier 2 Paraphrase | 4 | `sroie_283`, `284`, `292`, `295` | No | All satisfy dining receipt inquiry at Unihakka Johor. |

The complete record-by-record adjudication for all 400 instances is archived in [`data/human_queries_adjudicated.csv`](file:///Users/anuragkrishnavennela/Projects/MemoryVerse/data/human_queries_adjudicated.csv).

---

## 3. Disaggregated Retrieval Performance Across Policies

We re-evaluated Word BM25 and Character 3–5gram BM25 on the 813-document test corpus across all cohorts and relevance policies. Tie handling uses standard fractional reciprocal rank; missing results receive zero score ($\text{RR}=0$).

### Full Metric Results Table

| Evaluation Cohort | $N$ | Word MRR | Word Hit@1 | Word Hit@5 | Word Hit@10 | Char MRR | Char Hit@1 | Char Hit@5 | Char Hit@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Probes ($N=400$) — Strict Single-Target** | 400 | 0.5900 | 0.4850 | 0.6925 | 0.8025 | 0.5952 | 0.5000 | 0.6800 | 0.8050 |
| **All Probes ($N=400$) — Adjudicated Multi-Relevance** | 400 | **0.6740** | 0.5825 | 0.7950 | 0.8500 | **0.6825** | 0.6050 | 0.7600 | 0.8500 |
| **Unique Probes ($N=253$) — Single-Target** | 253 | **0.7599** | 0.6877 | 0.8340 | 0.8972 | **0.7658** | 0.7075 | 0.8300 | 0.8972 |
| **Ambiguous Repeated ($N=147$) — Single-Target** | 147 | 0.2976 | 0.1361 | 0.4490 | 0.6395 | 0.3016 | 0.1429 | 0.4218 | 0.6463 |
| **Ambiguous Repeated ($N=147$) — Adjudicated Multi-Relevance** | 147 | **0.5263** | 0.4014 | 0.7279 | 0.7687 | **0.5391** | 0.4286 | 0.6395 | 0.7687 |
| **Tier 1 Manual Probes ($N=200$) — Single-Target** | 200 | 0.6528 | 0.5500 | 0.7550 | 0.8550 | 0.6418 | 0.5500 | 0.7350 | 0.8350 |
| **Tier 1 Manual Probes ($N=200$) — Multi-Relevance** | 200 | 0.6816 | 0.5850 | 0.7850 | 0.8700 | 0.6713 | 0.5850 | 0.7600 | 0.8500 |
| **Tier 2 Templated Probes ($N=200$) — Single-Target** | 200 | 0.5273 | 0.4200 | 0.6300 | 0.7500 | 0.5486 | 0.4500 | 0.6250 | 0.7750 |
| **Tier 2 Templated Probes ($N=200$) — Multi-Relevance** | 200 | 0.6665 | 0.5800 | 0.8050 | 0.8300 | 0.6937 | 0.6250 | 0.7600 | 0.8500 |

---

## 4. Scientific IR Synthesis

1. **Quantification of the False-Negative Penalty:**
   On the 147 ambiguous repeated queries, strict single-target evaluation imposes an artificial penalty of **$-0.2287$ MRR** on Word BM25 ($0.2976$ vs. $0.5263$) and **$-0.2375$ MRR** on Character BM25 ($0.3016$ vs. $0.5391$). The search engine frequently ranks a valid receipt from the same merchant at Rank 1, but is penalized because it is not the single arbitrary row-assigned receipt ID.

2. **True Performance on Unambiguous Queries:**
   On the 253 unique, non-duplicated queries, single-target retrieval is highly effective:
   - Word BM25: MRR = **0.7599**, Hit@10 = **0.8972**.
   - Character BM25: MRR = **0.7658**, Hit@10 = **0.8972**.

3. **Invariance of the Thesis:**
   Across all cohorts, tiers, and ground-truth policies:
   - Word and Character BM25 remain tightly coupled (differences $\le 0.027$ MRR).
   - In no human query regime does Character BM25 collapse below Word BM25 (as seen under synthetic `gpt-4o-mini` entity-stripped paraphrases where Char drops to 0.2427).
   - This proves that sub-word character indexing remains highly robust whenever natural entity anchors are preserved.

---

## 5. Adjudication Protocol and Provenance Metadata

- **Adjudicator:** Independent Information Retrieval & Reproducibility Auditor.
- **Evidence Inspected:** Full OCR transcriptions (Apple Vision, Tesseract 5.5, EasyOCR 1.7.2), ground-truth bounding box text, merchant entity names, transaction totals, street addresses, and line-item descriptions.
- **Decision Rule:** A candidate document is marked relevant if and only if it strictly satisfies all explicit geographical and entity constraints in the query text without contradicting any stated purchase details.

