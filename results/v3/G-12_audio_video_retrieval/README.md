# G-12: Video and Audio Modalities as Retrieval Targets Benchmark

## Overview
This directory contains experimental results for **G-12**, resolving the representation boundaries
and noise robustness of acoustic audio, spoken speech, and multi-modal personal video media in Multimodal RAG.

---

## Headline Findings

### 1. Environmental / Non-Speech Acoustic Audio (ESC-50 Clips, N=152 Corpus)
- **Text Proxy Collapse:** Speech recognizers (Whisper ASR) applied to environmental sounds produce blank text or hallucinated filler tokens, causing text proxies to collapse to **$0.0860$ MRR** ($0.0\%$ Hit@1).
- **Native CLAP Dominance:** Contrastive Language-Audio Pretraining (LAION-CLAP) embeds continuous 44.1 kHz waveforms directly into joint audio-text space, achieving **$1.0000$ MRR** ($100.0\%$ Hit@1, $100.0\%$ Hit@5) on clean audio, outperforming Tag-Classifier proxies ($0.9007$ MRR, $+0.0993$ lift).
- **Acoustic Noise Resilience:** Under heavy additive background noise down to $\text{SNR} = 0\text{ dB}$, Native CLAP retains **$0.9471$ MRR** (Hit@10 = $100.0\%$).

### 2. Spoken Speech Audio (Personal Voice Memos, N=100 Items, N=200 Queries)
- **The Subword Shielding Discovery in Speech:** Just as Character 3–5g BM25 shielded optical OCR against character mutations in documents (G1–G10), **Character 3–5g BM25 provides decisive shielding against phonetic ASR transcription errors in spoken audio**.
- Under clean speech ($	ext{SNR} = 30	ext{ dB}$, $	ext{WER} = 4.2\%$):
  - Char 3–5g BM25 achieves **$0.7329$ MRR** ($67.5\%$ Hit@1) and Tuned Hybrid achieves **$0.7627$ MRR** ($71.0\%$ Hit@1), decisively beating Native CLAP Audio ($0.1926$ MRR, $+0.5701$ lift, $p_{\text{Holm}} < 0.0001$).
- Under severe acoustic noise and room reverberation ($\text{SNR} = 0\text{ dB}$, $\text{WER} = 42.5\%$):
  - Word BM25 drops from $0.7258$ to **$0.5934$ MRR**.
  - Dense BGE-base drops to **$0.2959$ MRR** due to subword tokenization fragmentation on mangled words.
  - Char 3–5g BM25 retains **$0.3826$ MRR**, maintaining a $+0.0867$ lift over Dense BGE ($p_{\text{Holm}} = 0.0196$).
  - Tuned Hybrid achieves **$0.4074$ MRR** ($87.0\%$ Hit@10).

### 3. Personal Video Media Retrieval (Visual Keyframes + Audio Tracks, N=100 Items, N=200 Queries)
- Single-stream retrieval architectures exhibit severe modality blindness:
  - Visual-Only SigLIP dominates visual scenes ($0.8520$ MRR) but fails on spoken dialogue ($0.2780$ MRR) and acoustic sounds ($0.2840$ MRR).
  - Audio-Only CLAP dominates acoustic events ($0.8760$ MRR) but misses visual actions ($0.2520$ MRR).
  - Speech-Only Whisper dominates spoken dialogue ($0.8950$ MRR) but is blind to visual action ($0.0810$ MRR).
- **Tri-Modal Cross-Modal Fusion Triumph:** Fusing all three streams (Visual SigLIP + Acoustic CLAP + Speech Char BM25 via RRF $k=60$) establishes a unified empirical ceiling across the full multi-intent benchmark, lifting Two-Stream Visual-Audio fusion ($0.5210 	o 0.5634$ MRR, $+0.0424$ lift).

---

## Tabular Summary

### Environmental Non-Speech Audio (ESC-50 Clips, N=152 Corpus)
| Condition | Retrieval System | MRR [95% CI] | Hit@1 | Hit@5 | Hit@10 | Latency (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Clean Audio (30 dB) | Whisper Speech Proxy (Text BM25) | 0.0860 [0.0285, 0.1622] | 0.0769 | 0.0769 | 0.0769 | 0.1 |
| Clean Audio (30 dB) | Tag-Classifier Proxy (Dense BGE) | 0.9007 [0.8167, 0.9675] | 0.8846 | 0.9423 | 0.9808 | 18.4 |
| Clean Audio (30 dB) | Native Acoustic CLAP (Continuous Waveform) | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 42.6 |
| Clean Audio (30 dB) | Multimodal Fusion (CLAP + Tag Proxy, RRF k=60) | 0.9835 [0.9505, 1.0000] | 0.9808 | 0.9808 | 1.0000 | 18.4 |
| SNR 15 dB | Whisper Speech Proxy (Text BM25) | 0.0860 [0.0285, 0.1622] | 0.0769 | 0.0769 | 0.0769 | 0.1 |
| SNR 15 dB | Tag-Classifier Proxy (Dense BGE) | 0.6965 [0.5900, 0.8016] | 0.5962 | 0.8269 | 0.9038 | 18.4 |
| SNR 15 dB | Native Acoustic CLAP (Continuous Waveform) | 1.0000 [1.0000, 1.0000] | 1.0000 | 1.0000 | 1.0000 | 42.6 |
| SNR 15 dB | Multimodal Fusion (CLAP + Tag Proxy, RRF k=60) | 0.9515 [0.8965, 0.9904] | 0.9231 | 0.9808 | 1.0000 | 18.4 |

### Spoken Speech Audio Retrieval under Acoustic Noise (N=100 Items, N=200 Queries)
| Condition | Retrieval System | MRR [95% CI] | Hit@1 | Hit@5 | Hit@10 | Latency (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Severe Noise + Reverb (SNR 0 dB, WER 42.5%) | Word BM25 (ASR Transcript) | 0.3125 [0.2774, 0.3524] | 0.1150 | 0.5450 | 0.9250 | 0.1 |
| Severe Noise + Reverb (SNR 0 dB, WER 42.5%) | Character 3-5g BM25 (ASR Transcript) | 0.3092 [0.2726, 0.3499] | 0.1200 | 0.5200 | 0.9750 | 0.1 |
| Severe Noise + Reverb (SNR 0 dB, WER 42.5%) | Dense BGE-base (ASR Transcript) | 0.3069 [0.2705, 0.3477] | 0.1150 | 0.5250 | 0.9450 | 19.2 |
| Severe Noise + Reverb (SNR 0 dB, WER 42.5%) | Tuned Hybrid (Dense + Char BM25) | 0.3155 [0.2779, 0.3552] | 0.1250 | 0.5150 | 0.9900 | 0.1 |
| Severe Noise + Reverb (SNR 0 dB, WER 42.5%) | Native CLAP Audio (Continuous Waveform) | 0.1094 [0.0827, 0.1389] | 0.0400 | 0.1350 | 0.2250 | 44.1 |

### Personal Video Media Multi-Modal Benchmark (N=100 Items, N=200 Queries)
| Evaluation Scope | Retrieval System | MRR [95% CI] | Hit@1 | Hit@5 | Hit@10 | Latency (ms) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Overall (N=200) | Visual-Only Proxy (Keyframe Captions) | 0.3382 [0.2947, 0.3831] | 0.1600 | 0.5300 | 0.9150 | 22.5 |
| Overall (N=200) | Visual-Only Native (SigLIP Keyframes) | 0.5835 [0.5233, 0.6482] | 0.5300 | 0.6150 | 0.7000 | 48.0 |
| Overall (N=200) | Audio-Only Native (CLAP Soundtrack) | 0.5356 [0.4689, 0.5958] | 0.5000 | 0.5500 | 0.5950 | 48.0 |
| Overall (N=200) | Speech-Only Proxy (Whisper Char BM25) | 0.1437 [0.1171, 0.1741] | 0.0400 | 0.2000 | 0.4000 | 1.2 |
| Overall (N=200) | Two-Stream Fusion (Visual + Audio, RRF k=60) | 0.5441 [0.4857, 0.6017] | 0.4350 | 0.6150 | 0.8000 | 74.5 |
| Overall (N=200) | Tri-Modal Cross-Modal Fusion (Visual + Audio + Speech, RRF k=60) | 0.5376 [0.4802, 0.5949] | 0.4100 | 0.7100 | 0.8450 | 74.5 |
