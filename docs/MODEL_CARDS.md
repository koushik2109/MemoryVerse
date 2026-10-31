# Model cards

## CLIP / SigLIP

- Purpose: image-text or frame-text retrieval and zero-shot labels.
- Input: decoded image/frame and text query.
- Limitation: not a people identity system; performance varies by domain and
  language. Do not infer sensitive traits or emotional states from embeddings.
- Operational: checkpoint download, CPU/GPU memory, and latency must be
  recorded per deployment.
- License/version: record the exact Hugging Face checkpoint and license in the
  deployment manifest before shipping.

## BGE-M3

- Purpose: text embedding for captions, transcripts, and text queries.
- Limitation: its vectors are not directly comparable to CLIP/SigLIP vectors;
  combine modalities through separate retrieval and calibrated fusion.

## Whisper

- Purpose: speech transcription with timestamps where the selected wrapper
  supports them.
- Limitation: accents, noise, names, and private-language content can be
  misrecognized. Transcript confidence must be retained when available.

## Hosted LLM/TTS providers

Provider, model, region, data-retention policy, and consent must be recorded in
deployment configuration. Personal media must not be sent externally merely
because an API key is present.
