#!/usr/bin/env python3
"""
run_funsd_easyocr_replication.py: EasyOCR 1.7.2 Replication on FUNSD Forms.

Executes empirical evaluation of EasyOCR 1.7.2 on 199 FUNSD administrative forms
under both Clean OCR and Degraded Optical Blur, evaluating retrieval across
the 100 FUNSD probes + 500 COCO distractors across 5 retrieval architectures:
- BM25 (Tuned Word)
- BM25 (Char 3-5gram)
- Dense MiniLM (all-MiniLM-L6-v2)
- Dense BGE-base (bge-base-en-v1.5)
- Tuned Hybrid (Sparse + Dense)

Outputs:
- results/v3/funsd_easyocr_clean_ocr.json
- results/v3/funsd_easyocr_degraded_ocr.json
- results/v3/funsd_easyocr_replication.csv
"""

import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from transformers import AutoTokenizer, AutoModel
import easyocr

REPO_ROOT = Path(__file__).resolve().parents[1]
FUNSD_DIR = REPO_ROOT / "ai_engine" / "rag" / "data" / "funsd" / "dataset"
BENCHMARK_PATH = REPO_ROOT / "ai_engine" / "rag" / "data" / "v2" / "public_benchmark_v2.json"
PROBES_PATH = REPO_ROOT / "results" / "v3" / "funsd_probes.json"
V3_RESULTS_DIR = REPO_ROOT / "results" / "v3"

EASY_CLEAN_CACHE = V3_RESULTS_DIR / "funsd_easyocr_clean_ocr.json"
EASY_DEG_CACHE = V3_RESULTS_DIR / "funsd_easyocr_degraded_ocr.json"
EASY_REPLICATION_CSV = V3_RESULTS_DIR / "funsd_easyocr_replication.csv"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
B_BOOT = 2000

# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKENIZATION AND BM25 UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_words(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)]

def tokenize_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> List[str]:
    clean = re.sub(r"\s+", " ", text.lower().strip())
    if not clean:
        return []
    ngrams = []
    length = len(clean)
    for n in range(n_min, n_max + 1):
        if length >= n:
            for i in range(length - n + 1):
                ngrams.append(clean[i : i + n])
    return ngrams if ngrams else [clean]

class PurePythonBM25:
    def __init__(self, corpus: List[str], k1: float = 0.9, b: float = 0.5, char_ngram: bool = False):
        self.k1 = k1
        self.b = b
        self.char_ngram = char_ngram
        self.corpus_size = len(corpus)
        self.doc_lengths: List[int] = []
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        for doc_id, doc_text in enumerate(corpus):
            tokens = tokenize_char_ngrams(doc_text) if char_ngram else tokenize_words(doc_text)
            self.doc_lengths.append(len(tokens))
            for t in tokens:
                if t not in self.inverted_index:
                    self.inverted_index[t] = {}
                self.inverted_index[t][doc_id] = self.inverted_index[t].get(doc_id, 0) + 1
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.corpus_size)

    def score(self, query: str) -> np.ndarray:
        q_tokens = tokenize_char_ngrams(query) if self.char_ngram else tokenize_words(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for t in q_tokens:
            if t not in self.inverted_index:
                continue
            postings = self.inverted_index[t]
            df = len(postings)
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / max(1e-6, denom)
        return scores

class DenseModelRetriever:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
                pooled = self.mean_pooling(out, encoded["attention_mask"])
                normed = F.normalize(pooled, p=2, dim=1)
                embeddings.append(normed.cpu().numpy())
        return np.vstack(embeddings)

def compute_rr_and_hit10(scores: np.ndarray, target_idx: int) -> Tuple[float, float]:
    ranks = np.argsort(-scores)
    pos = np.where(ranks == target_idx)[0]
    if len(pos) == 0:
        return 0.0, 0.0
    rank = pos[0] + 1
    rr = float(1.0 / rank)
    h10 = float(1.0 if rank <= 10 else 0.0)
    return rr, h10

def cluster_bootstrap_ci(values: np.ndarray, cluster_ids: List[str], B: int = 2000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: np.where(np.array(cluster_ids) == c)[0] for c in unique_clusters}
    boot_means = []
    for _ in range(B):
        sampled = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx_list = []
        for c in sampled:
            idx_list.extend(cluster_to_idx[c])
        boot_means.append(np.mean(values[idx_list]))
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

# ─────────────────────────────────────────────────────────────────────────────
# 2. OPTICAL DEGRADATION FOR FUNSD FORMS
# ─────────────────────────────────────────────────────────────────────────────

def degrade_image(img: Image.Image, seed_key: str) -> Image.Image:
    """Applies realistic optical blur degradation: Gaussian blur sigma=1.5, 50% scale, skew, JPEG Q20."""
    img = img.convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    w, h = img.size
    img = img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BILINEAR)
    angle = 2.5 if (hash(seed_key) % 2 == 0) else -2.5
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    buf.seek(0)
    return Image.open(buf)

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXTRACTION WORKER WITH CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def run_easyocr_on_funsd(
    img_paths: List[Path],
    cache_path: Path,
    reader: easyocr.Reader,
    desc: str,
    degrade: bool = False,
) -> Dict[str, List[str]]:
    results = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} cached {desc} EasyOCR transcriptions from {cache_path}")
        except Exception as e:
            print(f"Warning: could not read cache {cache_path}: {e}")

    remaining = [p for p in img_paths if p.stem not in results]
    if not remaining:
        print(f"All {len(img_paths)} {desc} images already transcribed via EasyOCR.")
        return results

    print(f"Transcribing {len(remaining)} {desc} FUNSD images with EasyOCR 1.7.2...")
    t0 = time.time()
    for idx, p in enumerate(remaining):
        try:
            img = Image.open(p)
            if degrade:
                img = degrade_image(img, p.stem)
            arr = np.array(img.convert("RGB"))
            lines = reader.readtext(arr, detail=0)
            results[p.stem] = lines
        except Exception as e:
            print(f"Error reading {p.name}: {e}")
            results[p.stem] = []

        if (idx + 1) % 10 == 0 or (idx + 1) == len(remaining):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(1e-3, elapsed)
            rem_sec = (len(remaining) - (idx + 1)) / max(1e-3, rate)
            print(f"  [{desc}] {idx+1}/{len(remaining)} images done ({rate:.2f} img/s, ~{rem_sec/60:.1f}m left)")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Completed {desc} extraction in {time.time()-t0:.2f}s.")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN REPLICATION EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FUNSD EasyOCR 1.7.2 Empirical Replication Pipeline")
    print(f"Hardware: {DEVICE}")
    print("=" * 70)

    # Find all 199 FUNSD form images
    test_img_dir = FUNSD_DIR / "testing_data" / "images"
    train_img_dir = FUNSD_DIR / "training_data" / "images"
    all_img_paths = sorted(list(test_img_dir.glob("*.png")) + list(train_img_dir.glob("*.png")))
    print(f"Found {len(all_img_paths)} FUNSD images across testing ({len(list(test_img_dir.glob('*.png')))}) and training ({len(list(train_img_dir.glob('*.png')))}).")

    # Load FUNSD test probes and public benchmark distractors
    with open(PROBES_PATH, "r", encoding="utf-8") as f:
        probes_data = json.load(f)
    test_probes = probes_data["test_probes"]
    print(f"Loaded {len(test_probes)} held-out FUNSD test probes.")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    coco_distractors = [r for r in bench_data["records"] if r["type"] == "natural_image" and r["split"] == "test"][:500]
    print(f"Loaded {len(coco_distractors)} MS-COCO test distractors.")

    # Initialize EasyOCR
    print("\nInitializing EasyOCR 1.7.2 Reader with gpu=True...")
    t_init = time.time()
    reader = easyocr.Reader(["en"], gpu=True)
    print(f"EasyOCR Reader initialized in {time.time()-t_init:.2f}s.")

    # Run extraction: Clean and Degraded
    easy_clean_ocr = run_easyocr_on_funsd(all_img_paths, EASY_CLEAN_CACHE, reader, desc="Clean", degrade=False)
    easy_deg_ocr = run_easyocr_on_funsd(all_img_paths, EASY_DEG_CACHE, reader, desc="Degraded Blur", degrade=True)

    # Build FUNSD test corpus: 100 test forms + 500 COCO distractors
    # The target_ids in test_probes are 'funsd_<stem>'
    test_stems = [p["target_id"].replace("funsd_", "") for p in test_probes]
    
    # Target index for each probe is its position among unique test forms
    unique_test_forms = list(dict.fromkeys(test_stems))
    stem_to_idx = {s: i for i, s in enumerate(unique_test_forms)}
    target_indices = [stem_to_idx[s] for s in test_stems]
    cluster_ids = [p["target_id"] for p in test_probes]

    q_verb = [p["verbatim_query"] for p in test_probes]
    q_light = [p["light_paraphrase"] for p in test_probes]

    print("\nLoading sentence-transformers models on", DEVICE)
    minilm = DenseModelRetriever("sentence-transformers/all-MiniLM-L6-v2", DEVICE)
    bge = DenseModelRetriever("BAAI/bge-base-en-v1.5", DEVICE)

    print("Encoding queries...")
    q_verb_mini = minilm.encode(q_verb)
    q_light_mini = minilm.encode(q_light)
    q_verb_bge = bge.encode(q_verb)
    q_light_bge = bge.encode(q_light)

    # Corpus for each condition
    def build_corpus(ocr_dict: Dict[str, List[str]]) -> List[str]:
        corpus = []
        for s in unique_test_forms:
            lines = ocr_dict.get(s, [])
            corpus.append("\n".join(lines))
        for r in coco_distractors:
            corpus.append(r.get("ground_truth_text", ""))
        return corpus

    corpus_clean = build_corpus(easy_clean_ocr)
    corpus_deg = build_corpus(easy_deg_ocr)
    print(f"Corpus size: {len(corpus_clean)} ({len(unique_test_forms)} test forms + {len(coco_distractors)} distractors)")

    results_table = []

    for cond_name, doc_corpus in [("EasyOCR Clean", corpus_clean), ("EasyOCR Blur", corpus_deg)]:
        print(f"\nEvaluating Retrieval on {cond_name}...")
        bm25_word = PurePythonBM25(doc_corpus, k1=0.9, b=0.5, char_ngram=False)
        bm25_char = PurePythonBM25(doc_corpus, k1=0.9, b=0.5, char_ngram=True)
        print("  Encoding document corpus with dense models...")
        doc_emb_mini = minilm.encode(doc_corpus)
        doc_emb_bge = bge.encode(doc_corpus)

        for q_type, q_list, q_mini, q_bge in [
            ("verbatim", q_verb, q_verb_mini, q_verb_bge),
            ("light", q_light, q_light_mini, q_light_bge),
        ]:
            mrr_word, h10_word = [], []
            mrr_char, h10_char = [], []
            mrr_mini, h10_mini = [], []
            mrr_bge, h10_bge = [], []
            mrr_hyb, h10_hyb = [], []

            alpha = 0.2 if q_type == "verbatim" else 0.3

            for i, q in enumerate(q_list):
                t_idx = target_indices[i]

                # BM25 word
                s_w = bm25_word.score(q)
                rr, h = compute_rr_and_hit10(s_w, t_idx)
                mrr_word.append(rr); h10_word.append(h)

                # BM25 char
                s_c = bm25_char.score(q)
                rr, h = compute_rr_and_hit10(s_c, t_idx)
                mrr_char.append(rr); h10_char.append(h)

                # Dense MiniLM
                s_m = doc_emb_mini @ q_mini[i]
                rr, h = compute_rr_and_hit10(s_m, t_idx)
                mrr_mini.append(rr); h10_mini.append(h)

                # Dense BGE
                s_b = doc_emb_bge @ q_bge[i]
                rr, h = compute_rr_and_hit10(s_b, t_idx)
                mrr_bge.append(rr); h10_bge.append(h)

                # Tuned Hybrid
                norm_w = (s_w - s_w.min()) / max(1e-6, s_w.max() - s_w.min())
                norm_b = (s_b - s_b.min()) / max(1e-6, s_b.max() - s_b.min())
                s_hyb = (1.0 - alpha) * norm_w + alpha * norm_b
                rr, h = compute_rr_and_hit10(s_hyb, t_idx)
                mrr_hyb.append(rr); h10_hyb.append(h)

            for sys_name, mrr_list, h10_list in [
                ("BM25 (Tuned)", mrr_word, h10_word),
                ("BM25 (Char 3-5gram)", mrr_char, h10_char),
                ("Dense MiniLM", mrr_mini, h10_mini),
                ("Dense BGE-base", mrr_bge, h10_bge),
                ("Tuned Hybrid", mrr_hyb, h10_hyb),
            ]:
                arr = np.array(mrr_list)
                ci_l, ci_u = cluster_bootstrap_ci(arr, cluster_ids, B=B_BOOT)
                mrr_mean = float(np.mean(arr))
                h10_mean = float(np.mean(h10_list))
                results_table.append({
                    "Condition": cond_name,
                    "Query_Type": q_type,
                    "System": sys_name,
                    "MRR": round(mrr_mean, 4),
                    "MRR_CI_Lower": round(ci_l, 4),
                    "MRR_CI_Upper": round(ci_u, 4),
                    "Hit10": round(h10_mean, 4),
                })
                print(f"  [{cond_name}] {q_type:<8} {sys_name:<20} MRR={mrr_mean:.4f} [{ci_l:.3f}, {ci_u:.3f}] Hit@10={h10_mean:.4f}")

    # Write CSV
    with open(EASY_REPLICATION_CSV, "w", encoding="utf-8") as f:
        f.write("Condition,Query_Type,System,MRR,MRR_CI_Lower,MRR_CI_Upper,Hit10\n")
        for r in results_table:
            f.write(f"{r['Condition']},{r['Query_Type']},{r['System']},{r['MRR']},{r['MRR_CI_Lower']},{r['MRR_CI_Upper']},{r['Hit10']}\n")
    print(f"\nSaved empirical FUNSD EasyOCR results to {EASY_REPLICATION_CSV}")

    # Format LaTeX columns
    print("\n" + "=" * 70)
    print("FORMATTED LATEX ROWS FOR TABLE X(b):")
    print("=" * 70)
    for q_type in ["verbatim", "light"]:
        print(f"\n% --- {q_type.upper()} ---")
        for sys_name in ["BM25 (Tuned)", "BM25 (Char 3-5gram)", "Dense MiniLM", "Dense BGE-base", "Tuned Hybrid"]:
            row_clean = next(r for r in results_table if r["Condition"] == "EasyOCR Clean" and r["Query_Type"] == q_type and r["System"] == sys_name)
            row_blur = next(r for r in results_table if r["Condition"] == "EasyOCR Blur" and r["Query_Type"] == q_type and r["System"] == sys_name)
            c_mrr, c_l, c_u = row_clean['MRR'], row_clean['MRR_CI_Lower'], row_clean['MRR_CI_Upper']
            b_mrr, b_l, b_u = row_blur['MRR'], row_blur['MRR_CI_Lower'], row_blur['MRR_CI_Upper']
            clean_str = f"\\makecell{{{c_mrr:.4f}\\\\ {{\\tiny [{c_l:.3f}, {c_u:.3f}]}}}}"
            blur_str = f"\\makecell{{{b_mrr:.4f}\\\\ {{\\tiny [{b_l:.3f}, {b_u:.3f}]}}}}"
            print(f" & {sys_name:<20} & ... & {clean_str} & {blur_str} \\\\")

if __name__ == "__main__":
    main()
