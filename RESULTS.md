# Empirical Evaluation Results & Decision Report

**Paper:** Multi-Stage Multimodal RAG with Text-Proxy Hybrid Retrieval (RAMP-RAG)  
**Evaluation Protocol:** Strict 50/50 Dev/Test partition by Image ID, zero target leakage, observable reliability (zero oracle), multi-seed robustness (Seeds 42, 43, 44), 2,000 bootstrap resamples, exact Clopper-Pearson intervals for saturated rates, paired permutation tests with step-down Holm-Bonferroni correction, standardized 3-run latency profiling on warm cache.

---

## 1. Core Decision

### **Does RAMP-RAG + clean stream beat unweighted hybrid at 20% and 40% noise with non-overlapping CIs and a Holm-corrected $p < 0.05$?**

### **PLAIN ANSWER: NO.**

### Rigorous Empirical Breakdown:

1. **At 20% OCR Character Noise:**
   - **Unweighted Convex Hybrid ($\alpha=0.7$):**
     - Hit@10 = $0.9636$, $95\%\text{ CI } [0.9333, 0.9879]$
     - MRR = $0.8254$, $95\%\text{ CI } [0.7769, 0.8722]$
   - **RAMP-RAG + Clean Stream ($\gamma^* = 0.80$):**
     - Hit@10 = $0.9939$, $95\%\text{ CI } [0.9758, 1.0000]$
     - MRR = $0.8624$, $95\%\text{ CI } [0.8210, 0.9003]$
   - **Confidence Interval Overlap:** **YES** (The $95\%$ bootstrap CIs $[0.7769, 0.8722]$ and $[0.8210, 0.9003]$ overlap across the range $[0.8210, 0.8722]$).
   - **Statistical Significance:** **NOT SIGNIFICANT** ($p_{\text{Holm}} = 0.3252 > 0.05$).

2. **At 40% OCR Character Noise:**
   - **Unweighted Convex Hybrid ($\alpha=0.7$):**
     - Hit@10 = $0.8606$, $95\%\text{ CI } [0.8059, 0.9091]$
     - MRR = $0.6677$, $95\%\text{ CI } [0.6048, 0.7271]$
   - **RAMP-RAG + Clean Stream ($\gamma^* = 0.80$):**
     - Hit@10 = $0.9576$, $95\%\text{ CI } [0.9273, 0.9879]$
     - MRR = $0.7802$, $95\%\text{ CI } [0.7263, 0.8306]$
   - **Confidence Interval Overlap:** **YES** (The MRR CIs $[0.6048, 0.7271]$ and $[0.7263, 0.8306]$ touch/overlap at $[0.7263, 0.7271]$).
   - **Statistical Significance:** Statistically significant improvement ($p_{\text{Holm}} = 0.0084 < 0.05$) and non-overlapping Hit@10 CIs, but fails the joint criterion of non-overlapping MRR CIs across both 20% and 40% noise levels.

3. **Key Finding — The Lexical Dominance Phenomenon:**
   - Pure **BM25 significantly outperforms ALL hybrid and dense models** under character corruption:
     - At 20% Noise: BM25 achieves MRR = **$0.9473$** vs. Hybrid $0.8254$ and RAMP-RAG $0.8624$ ($p = 0.0006$).
     - At 40% Noise: BM25 achieves MRR = **$0.8993$** vs. Hybrid $0.6677$ and RAMP-RAG $0.7802$ ($p = 0.0006$).
   - **Root Cause:** Dense sentence embeddings (`all-MiniLM-L6-v2`) catastrophically degrade when characters are corrupted (MRR collapses from $0.8527$ at 0% noise to $0.5306$ at 20% and $0.3691$ at 40%). Incorporating a corrupt dense stream dilutes exact lexical matches unless the dense weight is aggressively zeroed.

---

## 2. Mandatory Claims to Remove or Correct in the Paper

Because the data demonstrates that RAMP-RAG + clean stream does **not** satisfy non-overlapping CIs and $p < 0.05$ over unweighted hybrid across both 20% and 40% noise, and because pure BM25 outperforms all proxy-fused methods under noise, the following claims **must be removed or corrected**:

| Inaccurate / Unsupported Claim | Why It Must Be Removed | Replacement / Data-Supported Framing |
|---|---|---|
| *"improves robustness against corrupted proxy text"* | Overlapping CIs with hybrid at 20% noise ($p=0.3252$); pure BM25 achieves superior robustness ($0.9473$ vs $0.8624$). | Replace with an honest failure mode analysis: character noise degrades dense text proxies faster than inverted indices, making lexical BM25 the most robust single stream. |
| *"+41.1% relative MRR under 20% proxy noise"* | The measured MRR for RAMP-RAG + clean stream at 20% noise is $0.8624$ vs. Hybrid $0.8254$ (+4.5% relative gain, statistically indistinguishable, $p=0.3252$). Original RAMP-RAG scored $0.6726$ (-18.5%). | Remove the +41.1% figure entirely from Abstract, Section IV, and Conclusion. Replace with measured values: $+4.5\%$ relative difference ($p=0.3252$). |
| *"without cascading latency penalties"* | The Cascade Ranker increases mean latency from $4.30$ ms (Hybrid) to $17.80$ ms (RAMP-RAG) and $39.56$ ms (full Cross-Encoder), a 4.1$\times$ latency increase. | Acknowledge that confidence-triggered escalation introduces a measurable latency trade-off ($17.80$ ms vs $4.30$ ms for fixed hybrid). |
| *"balances accuracy and latency across modalities"* | On MS-COCO with realistic BLIP proxies, direct visual grounding (CLIP zero-shot on raw pixels) achieves Hit@10 = $0.7600$ and MRR = $0.5045$, outperforming text-proxy hybrid ($0.6480$ / $0.3581$). | State plainly that text-proxy RAG is advantageous for explicit scene text and documents, but direct vision-language representations remain superior for pure imagery. |

---

## 3. Summary of Experimental Results Across All Tasks

### Task 2: Noise Robustness Sweep (Held-Out Test Split, Seeds 42, 43, 44)

| Noise | Method | Hit@1 | Hit@5 | Hit@10 [95% CI] | MRR [95% CI] | NDCG@10 | $p_{\text{Holm}}$ vs. Hybrid |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0%** | BM25 | 0.9636 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9818 [0.967, 0.994] | 0.9870 | 0.2509 |
| | Dense | 0.7697 | 0.9394 | 0.9636 [0.933, 0.988] | 0.8527 [0.804, 0.898] | 0.8809 | 0.0006 |
| | Hybrid (fixed $\alpha=0.7$) | 0.9818 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9909 [0.979, 1.000] | 0.9934 | Baseline |
| | Hybrid + Router | 0.8848 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9409 [0.914, 0.965] | 0.9566 | 0.0006 |
| | Original RAMP-RAG | 0.8303 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9053 [0.866, 0.940] | 0.9312 | 0.0006 |
| | RAMP-RAG + Clean Stream | 0.8909 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9424 [0.916, 0.966] | 0.9577 | 0.0006 |
| | Cross-Encoder | 0.9091 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9515 [0.923, 0.977] | 0.9644 | 0.0068 |
| **10%** | BM25 | 0.9152 | 0.9818 | 0.9818 [0.958, 1.000] | 0.9502 [0.920, 0.976] | 0.9592 | 0.7523 |
| | Dense | 0.5939 | 0.9091 | 0.9455 [0.909, 0.976] | 0.7394 [0.681, 0.795] | 0.7936 | 0.0006 |
| | Hybrid (fixed $\alpha=0.7$) | 0.8909 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9457 [0.918, 0.970] | 0.9600 | Baseline |
| | Hybrid + Router | 0.8061 | 0.9939 | 1.0000 [0.978, 1.000] | 0.8994 [0.861, 0.933] | 0.9255 | 0.0060 |
| | Original RAMP-RAG | 0.7333 | 0.9758 | 0.9939 [0.982, 1.000] | 0.8487 [0.802, 0.891] | 0.8872 | 0.0006 |
| | RAMP-RAG + Clean Stream | 0.8182 | 1.0000 | 1.0000 [0.978, 1.000] | 0.9058 [0.870, 0.938] | 0.9304 | 0.0620 |
| | Cross-Encoder | 0.7939 | 1.0000 | 1.0000 [0.978, 1.000] | 0.8909 [0.854, 0.925] | 0.9203 | 0.0243 |
| **20%** | BM25 | 0.9152 | 0.9758 | 0.9758 [0.952, 0.994] | 0.9473 [0.917, 0.974] | 0.9554 | 0.0006 |
| | Dense | 0.3697 | 0.7091 | 0.7515 [0.685, 0.812] | 0.5306 [0.467, 0.592] | 0.5846 | 0.0006 |
| | Hybrid (fixed $\alpha=0.7$) | 0.7030 | 0.9455 | 0.9636 [0.933, 0.988] | 0.8254 [0.777, 0.872] | 0.8610 | Baseline |
| | Hybrid + Router | 0.5636 | 0.8606 | 0.8970 [0.848, 0.939] | 0.7178 [0.663, 0.774] | 0.7634 | 0.0006 |
| | Original RAMP-RAG | 0.5273 | 0.8121 | 0.8424 [0.788, 0.897] | 0.6726 [0.607, 0.733] | 0.7161 | 0.0006 |
| | RAMP-RAG + Clean Stream | 0.7515 | 0.9818 | 0.9939 [0.976, 1.000] | 0.8624 [0.821, 0.900] | 0.8977 | 0.3252 |
| | Cross-Encoder | 0.7333 | 0.9394 | 0.9636 [0.933, 0.988] | 0.8345 [0.788, 0.878] | 0.8687 | 0.6933 |
| **40%** | BM25 | 0.8364 | 0.9576 | 0.9576 [0.927, 0.988] | 0.8993 [0.858, 0.939] | 0.9157 | 0.0006 |
| | Dense | 0.2000 | 0.5212 | 0.6545 [0.582, 0.727] | 0.3691 [0.313, 0.426] | 0.4392 | 0.0006 |
| | Hybrid (fixed $\alpha=0.7$) | 0.4909 | 0.8303 | 0.8606 [0.806, 0.909] | 0.6677 [0.605, 0.727] | 0.7174 | Baseline |
| | Hybrid + Router | 0.3636 | 0.7273 | 0.7758 [0.709, 0.836] | 0.5571 [0.497, 0.618] | 0.6133 | 0.0006 |
| | Original RAMP-RAG | 0.3394 | 0.6545 | 0.7030 [0.630, 0.770] | 0.5045 [0.439, 0.571] | 0.5546 | 0.0006 |
| | RAMP-RAG + Clean Stream | 0.6182 | 0.9212 | 0.9576 [0.927, 0.988] | 0.7802 [0.726, 0.831] | 0.8286 | 0.0084 |
| | Cross-Encoder | 0.4121 | 0.7879 | 0.8606 [0.806, 0.909] | 0.6066 [0.548, 0.666] | 0.6728 | 0.0178 |

---

### Task 3: 5-Row Component Ablation

| Configuration | COCO Test Hit@10 | COCO Test MRR | Clean Scene-Text Hit@10 | Clean Scene-Text MRR | 20% Noise Scene-Text Hit@10 [95% CI] | 20% Noise Scene-Text MRR [95% CI] | $p$ vs. Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Fixed $\alpha=0.7$)** | 0.6360 | 0.3767 | 1.0000 | 0.9909 | 0.9636 [0.933, 0.988] | 0.8254 [0.777, 0.872] | Baseline |
| **+ Router (Learned $\alpha_q$)** | 0.6480 | 0.3975 | 1.0000 | 0.9409 | 0.8970 [0.848, 0.939] | 0.7178 [0.663, 0.774] | 0.0001 |
| **+ Cascade ($\tau=0.065$)** | 0.6360 | 0.3697 | 1.0000 | 0.9909 | 0.9636 [0.933, 0.988] | 0.8442 [0.796, 0.890] | 0.0652 |
| **+ Reliability ($r_m$)** | 0.6480 | 0.3975 | 1.0000 | 0.9409 | 0.8424 [0.788, 0.897] | 0.6383 [0.573, 0.704] | 0.0001 |
| **Full RAMP-RAG (+ Clean Stream)** | 0.6480 | 0.3821 | 1.0000 | 0.9727 | 0.9939 [0.976, 1.000] | 0.8624 [0.821, 0.900] | 0.1626 |

---

### Task 4: MS-COCO with Realistic BLIP Proxies & Stronger Baselines

| Method / Configuration | Hit@1 | Hit@5 | Hit@10 [95% CI] | MRR [95% CI] | NDCG@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| BM25 (BLIP Proxy) | 0.1360 | 0.3160 | 0.4200 [0.356, 0.480] | 0.2325 [0.190, 0.276] | 0.2764 |
| Dense `all-MiniLM-L6-v2` (BLIP Proxy) | 0.2360 | 0.5400 | 0.6920 [0.636, 0.744] | 0.3873 [0.338, 0.435] | 0.4601 |
| Dense `BAAI/bge-base-en-v1.5` (BLIP Proxy) | 0.2800 | 0.5280 | 0.6360 [0.576, 0.692] | 0.4002 [0.348, 0.450] | 0.4566 |
| **CLIP Zero-Shot Visual (Raw Pixels)** | **0.3800** | **0.6440** | **0.7600 [0.708, 0.812]** | **0.5045 [0.455, 0.555]** | **0.5670** |
| Hybrid Convex (MiniLM + BM25) | 0.2160 | 0.5080 | 0.6480 [0.588, 0.704] | 0.3581 [0.310, 0.405] | 0.4285 |
| Hybrid Convex (BGE-base + BM25) | 0.2240 | 0.5080 | 0.6160 [0.556, 0.676] | 0.3557 [0.307, 0.404] | 0.4187 |
| Full RAMP-RAG (BLIP + CLIP Visual) | 0.2440 | 0.5440 | 0.6840 [0.628, 0.736] | 0.3908 [0.342, 0.440] | 0.4614 |
| Cross-Encoder (MiniLM Top-5) | 0.2360 | 0.5120 | 0.6480 [0.588, 0.704] | 0.3734 [0.326, 0.423] | 0.4397 |

---

### Task 5: Standardized Latency Profile

*Scope = Query Encoding + Retrieval + Score Fusion + Reranking/Cascade. Warm Cache, 3 Runs.*

| Retrieval Architecture | Mean Latency (ms) | P95 Latency (ms) |
|:---|:---:|:---:|
| BM25 (Sparse Inverted Index) | 0.02 | 0.04 |
| Dense (`all-MiniLM-L6-v2`) | 4.38 | 6.24 |
| Dense (`BAAI/bge-base-en-v1.5`) | 13.17 | 13.72 |
| CLIP Zero-Shot Visual | 7.89 | 10.36 |
| Hybrid Convex (Fixed $\alpha=0.7$) | 4.30 | 6.40 |
| Original RAMP-RAG | 17.54 | 40.65 |
| Proposed RAMP-RAG (+ Clean Stream) | 17.80 | 40.77 |
| Cross-Encoder (100% Cascade) | 39.56 | 54.40 |
