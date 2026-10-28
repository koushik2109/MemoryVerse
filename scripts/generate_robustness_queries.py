#!/usr/bin/env python3
"""
generate_robustness_queries.py
---------------------------------
[USER-RUN SCRIPT FOR TASK 3.1]
Regenerates light-paraphrase and answerable queries with a SECOND generator
(an open-weight LLM, e.g. Qwen2.5-7B-Instruct or Llama-3-8B-Instruct via vLLM/Ollama/HuggingFace),
and generates a CSV template for 100 human-authored evaluation queries.

Usage:
    python scripts/generate_robustness_queries.py --mode template
    python scripts/generate_robustness_queries.py --mode llm --model Qwen/Qwen2.5-7B-Instruct
    python scripts/generate_robustness_queries.py --mode api --api-base http://localhost:8000/v1 --model meta-llama/Llama-3-8B-Instruct
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSE_QUERIES_PATH = os.path.join(BASE_DIR, "results", "v6", "dose_response_queries_v6.json")
OUTPUT_GEN2_PATH = os.path.join(BASE_DIR, "data", "robustness_queries_gen2.json")
OUTPUT_HUMAN_TEMPLATE_PATH = os.path.join(BASE_DIR, "data", "human_query_template.csv")


def generate_human_template(num_queries: int = 100):
    """Generates a CSV template for human annotation across the first num_queries test probes."""
    if not os.path.exists(DOSE_QUERIES_PATH):
        print(f"Error: Base query file not found at {DOSE_QUERIES_PATH}")
        sys.exit(1)

    with open(DOSE_QUERIES_PATH, "r", encoding="utf-8") as f:
        probes = json.load(f)

    os.makedirs(os.path.dirname(OUTPUT_HUMAN_TEMPLATE_PATH), exist_ok=True)

    rows = []
    for item in probes[:num_queries]:
        probe_id = item.get("probe_id", "")
        target_id = item.get("target_id", "")
        verbatim = item.get("band_100", {}).get("query", "")
        gen1_paraphrase = item.get("band_40", {}).get("query", "")

        rows.append({
            "probe_id": probe_id,
            "target_id": target_id,
            "verbatim_query": verbatim,
            "gpt4o_mini_paraphrase": gen1_paraphrase,
            "human_light_paraphrase": "",  # To be filled by human annotator
            "human_answerable_question": "",  # To be filled by human annotator
            "annotator_id": "",
            "validation_notes": ""
        })

    with open(OUTPUT_HUMAN_TEMPLATE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "probe_id", "target_id", "verbatim_query", "gpt4o_mini_paraphrase",
            "human_light_paraphrase", "human_answerable_question",
            "annotator_id", "validation_notes"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Generated human query template with {len(rows)} rows at: {OUTPUT_HUMAN_TEMPLATE_PATH}")


def generate_with_transformers(model_name: str, max_samples: int = None):
    """Generates second-generator queries using HuggingFace Transformers pipeline."""
    try:
        import torch
        from transformers import pipeline
    except ImportError:
        print("Error: transformers and torch must be installed to run local open-weight generation.")
        print("Run: pip install transformers torch accelerate")
        sys.exit(1)

    print(f"Loading open-weight generator model: {model_name}...")
    generator = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        max_new_tokens=128,
        temperature=0.2,
        do_sample=True
    )

    with open(DOSE_QUERIES_PATH, "r", encoding="utf-8") as f:
        probes = json.load(f)

    subset = probes[:max_samples] if max_samples else probes
    results = []

    for i, item in enumerate(subset):
        verbatim = item.get("band_100", {}).get("query", "")
        prompt_para = (
            f"You are evaluating a document retrieval system. Given the exact text from a receipt:\n"
            f"\"{verbatim}\"\n"
            f"Write a natural, light paraphrase that a user might search to find this receipt. "
            f"Substitute synonyms or generic terms (e.g., 'store payment' instead of company name). "
            f"Output ONLY the query string:"
        )
        prompt_qa = (
            f"Given the receipt snippet:\n"
            f"\"{verbatim}\"\n"
            f"Write a natural, fact-seeking answerable question that can be answered by this receipt. "
            f"Output ONLY the question string:"
        )

        out_para = generator(prompt_para)[0]["generated_text"].split("Output ONLY the query string:")[-1].strip()
        out_qa = generator(prompt_qa)[0]["generated_text"].split("Output ONLY the question string:")[-1].strip()

        results.append({
            "probe_id": item.get("probe_id"),
            "target_id": item.get("target_id"),
            "verbatim_query": verbatim,
            "generator2_model": model_name,
            "gen2_light_paraphrase": out_para.strip('"\n '),
            "gen2_answerable_question": out_qa.strip('"\n ')
        })
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(subset)} queries...")

    with open(OUTPUT_GEN2_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Generated {len(results)} second-generator queries saved to {OUTPUT_GEN2_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Generate robustness validation queries.")
    parser.add_argument("--mode", choices=["template", "llm", "api"], default="template",
                        help="Mode: 'template' for human annotation CSV, 'llm' for local transformers.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or API model name.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of queries processed.")
    args = parser.parse_args()

    if args.mode == "template":
        generate_human_template(num_queries=100)
    elif args.mode == "llm":
        generate_with_transformers(args.model, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

