"""
Multimodal Fusion Module
Implements Reciprocal Rank Fusion (RRF) and weighted score fusion
across visual, textual, and acoustic retrieval channels.
"""
from typing import List, Dict, Any, Optional


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    r"""
    Applies Reciprocal Rank Fusion (RRF) across multiple ranked lists:
        RRF_score(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}
    Returns a unified sorted list with merged 'rrf_score'.
    """
    scores: Dict[str, float] = {}
    items_by_id: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = str(item.get(id_key, f"item_{rank}"))
            if item_id not in items_by_id:
                items_by_id[item_id] = dict(item)
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank + 1))

    # Sort items by accumulated RRF score descending
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for item_id in sorted_ids:
        entry = items_by_id[item_id]
        entry["rrf_score"] = round(scores[item_id], 5)
        fused_results.append(entry)

    return fused_results


def weighted_multimodal_fusion(
    items: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Fuses visual, text, and acoustic sub-scores based on dynamic query weights:
        final_score = w_visual * visual_sim + w_text * text_sim + w_audio * audio_sim
    """
    w = weights or {"visual": 0.45, "text": 0.35, "audio": 0.20}
    w_vis = w.get("visual", 0.45)
    w_txt = w.get("text", 0.35)
    w_aud = w.get("audio", 0.20)

    fused = []
    for it in items:
        entry = dict(it)
        vis_score = float(entry.get("visual_score", entry.get("similarity", 0.5)))
        txt_score = float(entry.get("text_score", entry.get("similarity", 0.5)))
        aud_score = float(entry.get("audio_score", 0.5))

        composite = (w_vis * vis_score) + (w_txt * txt_score) + (w_aud * aud_score)
        entry["fused_score"] = round(composite, 4)
        fused.append(entry)

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    return fused
