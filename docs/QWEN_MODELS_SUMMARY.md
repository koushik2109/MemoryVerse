# Qwen Models for MemoryVerse - Quick Reference

> **Status:** Proposed model mapping. The current agent implementations are
> deterministic Python services; the standalone video captioner uses the
> configured Ollama Qwen VLM. The mappings and performance figures below are
> not evidence of a deployed five-model Qwen stack.

## Agent-to-Model Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│                   MEMORYVERSE AGENTS                            │
└─────────────────────────────────────────────────────────────────┘

Agent 1: PLANNER
├── Base Model: Qwen/Qwen2.5-7B-Instruct
├── Size: 7.6B parameters
├── Task: Parse natural language → structured retrieval plans
├── LoRA Rank: 16
└── Trainable Params: ~20M (0.26%)

Agent 2: SCORER  
├── Base Model: Qwen/Qwen2-VL-7B-Instruct
├── Size: 8.3B parameters (vision + language)
├── Task: Evaluate photo quality + emotional significance
├── LoRA Rank: 32
└── Trainable Params: ~45M (0.54%)

Agent 3: STORYTELLER
├── Base Model: Qwen/Qwen2.5-14B-Instruct
├── Size: 14.7B parameters
├── Task: Generate creative narratives
├── LoRA Rank: 64
├── Training: DPO (Direct Preference Optimization)
└── Trainable Params: ~85M (0.58%)

Agent 4: AUDITOR
├── Base Model: Qwen/Qwen2.5-7B-Instruct (shared with Planner)
├── Size: 7.6B parameters  
├── Task: Verify claims against Evidence Manifest
├── LoRA Rank: 16
└── Trainable Params: ~20M (0.26%)

Agent 5: DIRECTOR
├── Base Model: Qwen/Qwen2-VL-7B-Instruct (shared with Scorer)
├── Size: 8.3B parameters
├── Task: Generate shot lists with camera movements
├── LoRA Rank: 48
└── Trainable Params: ~55M (0.66%)
```

---

## Why Qwen Over Alternatives?

| Feature | Qwen2.5/Qwen2-VL | Llama 3.1 | GPT-4 API | Gemini API |
|---------|------------------|-----------|-----------|------------|
| **Open Source** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Vision-Language** | ✅ Qwen2-VL | ❌ No native | ✅ GPT-4V | ✅ Gemini Pro Vision |
| **Fine-tunable** | ✅ Full control | ✅ Full control | ❌ Limited | ❌ Limited |
| **Context Length** | ✅ 128K | ⚠️ 32K-128K | ✅ 128K | ✅ 1M+ |
| **Multilingual** | ✅ 29+ languages | ⚠️ Limited | ✅ Good | ✅ Good |
| **Cost** | ✅ $0 (self-host) | ✅ $0 (self-host) | ❌ $0.01/1K tokens | ❌ $0.0025/1K tokens |
| **Privacy** | ✅ 100% local | ✅ 100% local | ❌ Cloud | ❌ Cloud |
| **LoRA Support** | ✅ Excellent | ✅ Excellent | ❌ N/A | ❌ N/A |
| **Commercial License** | ✅ Apache 2.0 | ✅ Llama 3 License | ❌ Proprietary | ❌ Proprietary |
| **Performance** | ✅ SOTA on many tasks | ✅ SOTA | ✅ SOTA | ✅ SOTA |

---

## Resource Requirements

### Development (Single User Testing)
```
Hardware: 1× RTX 4090 (24GB) or M3 Max (48GB)
Models: All 5 agents loaded
VRAM: ~22GB (bf16) or ~8GB (4-bit quantized)
Inference: Sequential agent calls, ~2min per video
Cost: $1,600 - $3,200 one-time
```

### Production (100+ Users)
```
Hardware: 1× NVIDIA A100 (80GB)
Models: All base models + 100× LoRA adapters in memory
VRAM: ~70GB
Throughput: 80 videos/hour
Cost: $3.67/hour (AWS p4d.24xlarge)
```

### Edge Deployment (Privacy-First)
```
Hardware: Apple M3 Max / RTX 4060 Ti
Models: Quantized 4-bit models
VRAM: ~12GB
Inference: ~4min per video
Cost: $2,000 - $2,500 one-time
```

---

## Model Files & Storage

### Base Models (Downloaded Once)
```
Qwen2.5-7B-Instruct:      15.2 GB
Qwen2.5-14B-Instruct:     29.4 GB
Qwen2-VL-7B-Instruct:     17.8 GB
─────────────────────────────────
Total Base Models:        62.4 GB
```

### LoRA Adapters (Per User, Per Agent)
```
Planner LoRA:      ~12 MB
Scorer LoRA:       ~15 MB
Storyteller LoRA:  ~18 MB
Auditor LoRA:      ~12 MB
Director LoRA:     ~15 MB
─────────────────────────────────
Total per user:    ~72 MB

For 1,000 users:   72 GB
For 10,000 users:  720 GB
```

**Comparison:**
- Without LoRA: 62.4 GB × 5 agents × 10,000 users = **3.1 PB** ❌
- With LoRA: 62.4 GB + 720 GB = **782 GB** ✅ (3,900× reduction!)

---

## Training Time & Cost

### Initial LoRA Training (From Scratch)
```
Dataset: 100 user interactions
Hardware: 1× A100 (40GB)
Time per agent: 2-4 hours
Total time (5 agents): 10-20 hours
Cost: $20-40 (cloud) or free (self-hosted)
```

### Continuous Learning (Per User)
```
Trigger: Every 10 new interactions
Training time: 10-30 minutes
Happens: Async in background
Cost per update: $0.50 - $2.00 (cloud)
```

### DPO Fine-Tuning (Storyteller)
```
Dataset: 500 preference pairs
Hardware: 1× A100 (80GB)
Time: 6-12 hours
Epochs: 3
Cost: $30-60 (cloud)
```

---

## Inference Performance

### Latency Breakdown (Single Video Generation)

| Agent | Model | Task | Latency |
|-------|-------|------|---------|
| **Planner** | Qwen2.5-7B | Parse query → retrieval plan | 200ms |
| ↓ | | Database vector search | 500ms |
| **Scorer** | Qwen2-VL-7B | Evaluate 50 photos | 5s |
| ↓ | | Select top 15 photos | 50ms |
| **Storyteller** | Qwen2.5-14B | Generate narrative (300 words) | 8s |
| ↓ | | Text-to-speech synthesis | 3s |
| **Auditor** | Qwen2.5-7B | Verify narrative claims | 2s |
| ↓ | | (May trigger regeneration loop) | 0-10s |
| **Director** | Qwen2-VL-7B | Generate shot list (15 shots) | 4s |
| ↓ | | Video rendering (MoviePy) | 40s |
| **TOTAL** | | | **~63s** |

**With Parallel Execution:**
- Scorer + Auditor run simultaneously: -2s
- Final latency: **~61s per video**

---

## Quantization Trade-offs

### 4-bit Quantization (GPTQ/AWQ)
```
VRAM Reduction: 75% (30GB → 7.5GB)
Speed Impact: -20% to -30%
Quality Loss: <2% on most tasks
Best for: Edge deployment, multi-model hosting
```

### 8-bit Quantization (LLM.int8)
```
VRAM Reduction: 50% (30GB → 15GB)
Speed Impact: -10% to -15%
Quality Loss: <1% on most tasks
Best for: Single GPU hosting
```

### bfloat16 (Recommended)
```
VRAM: Full (30GB)
Speed: Baseline
Quality: No loss
Best for: Production, accuracy-critical tasks
```

---

## Example: Complete Video Generation Pipeline

```python
from qwen_agent_manager import QwenAgentManager

# Initialize all agents
agents = QwenAgentManager(device="cuda", use_quantization=False)

# User query
user_id = "user_12345"
query = "Create a 60-second video of our beach trip last summer"

# Agent 1: Planner
retrieval_plan = agents.generate(
    agent_name="planner",
    user_id=user_id,
    prompt=f"Parse query: {query}"
)
# Output: {temporal: [2025-06-01, 2025-08-31], location: "beach", concepts: ["ocean", "sunset"]}

# Retrieve photos from vector DB
photos = vector_db.search(retrieval_plan)  # 50 candidates

# Agent 2: Scorer (with images)
scored_photos = []
for photo in photos:
    score = agents.generate(
        agent_name="scorer",
        user_id=user_id,
        prompt="Evaluate this photo for video inclusion",
        images=photo.image
    )
    scored_photos.append((photo, score))

selected_photos = sorted(scored_photos, key=lambda x: x[1], reverse=True)[:15]

# Agent 3: Storyteller
narrative = agents.generate(
    agent_name="storyteller",
    user_id=user_id,
    prompt=f"""Create 60s narrative for beach trip.
    Photos: {[p.description for p in selected_photos]}
    Evidence: {evidence_manifest}"""
)

# Agent 4: Auditor
audit_result = agents.generate(
    agent_name="auditor",
    user_id=user_id,
    prompt=f"""Verify narrative against evidence:
    Narrative: {narrative}
    Evidence: {evidence_manifest}"""
)

if audit_result["hallucinations_detected"]:
    # Trigger regeneration loop
    narrative = agents.generate("storyteller", user_id, "Regenerate without hallucinations")

# Agent 5: Director (with images)
shot_list = agents.generate(
    agent_name="director",
    user_id=user_id,
    prompt=f"""Generate shot list for 60s video.
    Photos: {selected_photos}
    Narrative: {narrative}""",
    images=[p.image for p in selected_photos]
)

# Render video
video = render_video(selected_photos, shot_list, narrative, audio)
return video  # Ready in ~60 seconds!
```

---

## Key Advantages Summary

### 1. **Zero API Costs**
- Self-hosted models = $0 per request
- 10,000 videos/month: $0 (vs $1,000-$5,000 with OpenAI/Gemini APIs)

### 2. **Complete Privacy**
- All processing happens locally
- No data leaves user's infrastructure
- GDPR/HIPAA compliant by default

### 3. **Full Customization**
- Fine-tune on your exact use case
- Adjust model behavior without waiting for API updates
- Per-user personalization via LoRA

### 4. **No Rate Limits**
- Process unlimited requests
- No throttling during peak usage
- Predictable performance

### 5. **Multilingual Native**
- 29+ languages out-of-the-box
- No translation API needed
- Cultural context understanding

### 6. **Vision-Language Integration**
- Qwen2-VL processes images natively
- No separate CLIP/ViT pipeline needed
- Unified multimodal reasoning

---

## Getting Started

### Quick Start (5 minutes)
```bash
# Install dependencies
pip install transformers torch peft accelerate

# Download models
python download_qwen_models.py

# Run inference
python demo_agent_inference.py
```

### Production Deployment (1 day)
1. Set up GPU instance (A100 recommended)
2. Deploy base models to GPU memory
3. Configure LoRA adapter storage (S3/PostgreSQL)
4. Set up training pipeline for continuous learning
5. Monitor VRAM usage and latency

### Full Fine-Tuning (1 week)
1. Collect user feedback data (ratings, overrides, regenerations)
2. Generate training datasets for each agent
3. Train initial LoRA adapters (10-20 hours)
4. A/B test personalized vs. base models
5. Deploy continuous learning system

---

## Additional Resources

- **Qwen2.5 Paper**: [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
- **Qwen2-VL Paper**: [arXiv:2409.12191](https://arxiv.org/abs/2409.12191)
- **Hugging Face**: [Qwen Model Hub](https://huggingface.co/Qwen)
- **Official Repo**: [github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- **LoRA Paper**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **DPO Paper**: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
