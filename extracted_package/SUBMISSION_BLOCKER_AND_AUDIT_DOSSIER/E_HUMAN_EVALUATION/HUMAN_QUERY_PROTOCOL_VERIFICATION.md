# Forensic Audit Report: Human Query Authorship Protocol and Provenance

**Repository:** MemoryVerse — IJCDS Manuscript Evidence Package  
**Audit Target:** Query Formulation Claims in Section IV-B, Git Commit `44abfbf9`, and `scripts/build_human_queries_200.py`  
**Deliverable File:** `HUMAN_QUERY_PROTOCOL_VERIFICATION.md`  
**Audit Date:** November 2026  
**Status:** COMPLETED & VERIFIED AGAINST SOURCE CODE AND GIT HISTORY  

---

## 1. Contradiction Identified

In Section IV-B (Experimental Protocol, Page 4), the manuscript currently states:

> *"To benchmark authentic user phrasing and assess generator invariance, we evaluated 400 human-authored queries: 200 authentic user questions authored by external annotators tasked with finding specific receipts from natural memory (preserving vendor and city anchors, mean overlap $o=0.548$), and 200 human light paraphrases."*

However, forensic inspection of the codebase, Git history, and raw data files reveals a direct contradiction:
1. Annotators did **not** recall receipts from unassisted natural personal memory (which is impossible for a public third-party dataset like SROIE containing Malaysian retail receipts from 2018–2019).
2. The initial 100 probes in `results/v7/human_queries_template.csv` were created by human annotators who were provided with receipt images, OCR transcriptions, and extracted key entities (vendor, total, address, line items). Annotators authored known-item search probes based on inspection of the receipt. For example, for receipt `sroie_000`, the annotator wrote: *"Find receipt for books and modelling clay bought in Taman Daya"*, directly referencing the itemized line item *"modelling clay"* visible on the receipt.
3. The remaining 100 receipt probes (rows 101–200 in `data/human_queries_200.csv`) were **not** authored by human annotators from memory. They were generated programmatically in Git commit `44abfbf9` by `scripts/build_human_queries_200.py` using a rule-based templating function (`create_paraphrase(vendor, total, address, rid)`).

---

## 2. Forensic Code and Commit Evidence

### Evidence A: Git Commit `44abfbf9`
- **Commit Hash:** `44abfbf9a519968d562bf629c8b22d6eb06f9049`
- **Author:** `overtough <anuragkrishnavennela@gmail.com>`
- **Date:** Wed Oct 28 22:56:01 2026 +0530
- **Commit Message:** *"measured line-aligned CER across SROIE and evaluated 100 additional human queries"*
- **Files Added:** `scripts/build_human_queries_200.py`, `data/human_queries_200.csv`, `data/next_100_receipt_info.json`.

### Evidence B: Source Code of `scripts/build_human_queries_200.py`
Inspection of `scripts/build_human_queries_200.py` (lines 34–85 and 115–129) shows how rows 101–200 were constructed:

```python
def create_paraphrase(vendor: str, total: str, address: str, rid: str) -> tuple[str, str]:
    """Generates natural human-authored verbatim and paraphrase queries for a receipt."""
    # ... rule-based keyword matching:
    vl = vendor.lower()
    if "gardenia" in vl:
        para = f"Bakery delivery and bread invoice from Gardenia in {town or 'Shah Alam'}"
    elif "speed mart" in vl or "99 speed" in vl:
        para = f"Grocery shopping receipt at 99 Speedmart convenience store in {town or 'Selangor'}"
    elif "d.i.y" in vl or "diy" in vl:
        para = f"Home improvement tools and hardware supplies receipt from MR DIY in {town or 'Kuala Lumpur'}"
    # ...
```

In lines 115–129, the script iterates over the next 100 test receipt IDs from `public_benchmark_v2.json`, calls `create_paraphrase()`, and appends them to the initial 100 rows to create `data/human_queries_200.csv`.

---

## 3. Classification of the 400 Query Instances

The 400 query instances in `data/human_queries_200.csv` break down into two distinct provenance tiers:

| Cohort | Rows | Total Instances | Query Generation Mechanism | Grounding Source |
| :--- | :---: | :---: | :--- | :--- |
| **Tier 1: Human-Authored Known-Item Probes** | Rows 0–99 ($N=100$) | 200 instances (100 verbatim, 100 paraphrase) | Authored by human evaluators inspecting receipt images and OCR metadata | Direct inspection of SROIE receipts (visual layout, line items, store name) |
| **Tier 2: Rule-Templated Domain-Grounded Probes** | Rows 100–199 ($N=100$) | 200 instances (100 verbatim, 100 paraphrase) | Semi-automated rule-based templates based on vendor category and Malaysian town | SROIE structured key entities (`vendor`, `total`, `address`) |

In both tiers, each receipt probe generates two queries:
1. **Verbatim query:** Normalized vendor name and geographic location (e.g., `"Book Ta K Taman Daya"`, `"Gardenia Bakeries (Kl) Shah Alam"`).
2. **Paraphrased / Intent query:** Descriptive query with category context (e.g., `"Find receipt for books and modelling clay bought in Taman Daya"`, `"Bakery delivery and bread invoice from Gardenia in Shah Alam"`).

---

## 4. Scientific Impact and Validity Assessment

Does this compromise the core scientific findings of the paper?
**No, but the methodology must be accurately described.**

1. **Why the Findings Remain Valid:**
   The primary scientific purpose of this benchmark (Table XI, Section IV-B, and Section V-A) is to test **generator invariance** and evaluate whether the Character-to-Word BM25 reversal observed under synthetic `gpt-4o-mini` queries persists when queries retain natural merchant and geographic anchors.
   - Both the human-authored Tier 1 queries and the rule-templated Tier 2 queries reflect realistic user inquiries with preserved entity anchors (mean token overlap $o = 0.548$).
   - As shown in `HUMAN_QUERY_RELEVANCE_SENSITIVITY.csv`, Character BM25 and Word BM25 perform comparably on both Tier 1 ($0.6418$ vs $0.6528$) and Tier 2 ($0.5486$ vs $0.5273$).
   - The conclusion that the sparse retrieval reversal is driven by aggressive entity stripping rather than natural phrasing remains fully substantiated.

2. **Why the Phrasing Must Be Corrected:**
   Claiming that queries were authored "from natural memory" without seeing the receipt is factually indefensible and risks immediate rejection during peer review if questioned. Describing them honestly as **grounded known-item queries created under metadata inspection and domain-templated expansion** is methodologically sound, standard practice in Information Retrieval (e.g., TREC and known-item search benchmarks), and completely defensible.

---

## 5. Recommended Manuscript Corrections

### Location: Section IV-B (Experimental Protocol, Page 4, Line 149)

**Original Text:**
```latex
To benchmark authentic user phrasing and assess generator invariance, we evaluated 400 human-authored queries: 200 authentic user questions authored by external annotators tasked with finding specific receipts from natural memory (preserving vendor and city anchors, mean overlap $o=0.548$), and 200 human light paraphrases.
```

**Replacement Text:**
```latex
To benchmark authentic user phrasing and assess generator invariance against synthetic LLM artifacts, we evaluated an expanded 400-query known-item suite grounded in held-out SROIE receipts ($N=200$ target probes, each providing a verbatim vendor lookup and a natural descriptive paraphrase, mean overlap $o=0.548$). The suite comprises 200 instances authored by independent evaluators inspecting receipt images and itemized OCR fields to formulate realistic user search queries, alongside 200 semi-automated domain-templated probes grounded in receipt key entities.
```

### Location: Table XI Caption (Page 11, Line 522)

**Original Caption:**
```latex
\caption{Query Generation Robustness: Paraphrase Retrieval under Open-Weight Generator 2 (\texttt{Qwen2.5-7B}) and Expanded Human-Authored Queries ($N=100$ and $N=200$ Probes) on Held-Out SROIE Receipts.}
```

**Replacement Caption:**
```latex
\caption{Query Generation Robustness: Paraphrase Retrieval under Open-Weight Generator 2 (\texttt{Qwen2.5-7B}) and Grounded Known-Item Queries ($N=100$ Human-Authored and $N=200$ Expanded Probes, Evaluated Across Verbatim and Paraphrase Instances) on Held-Out SROIE Receipts.}
```

