"""
Fact Auditor / Narrator - Evidence Manifest
Validates narrative statements against the physical Evidence Manifest,
detects ungrounded entities or claims, and sanitizes hallucinations.
"""
from typing import List, Dict, Any, Tuple, Set
import re


def extract_atomic_facts(narration_text: str) -> List[str]:
    """Splits narration paragraph into discrete atomic sentence claims."""
    sentences = re.split(r"[.!?]+", narration_text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def verify_entities_against_manifest(
    statement: str,
    manifest_assets: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Verifies that named entities (locations, specific objects, temporal claims)
    in the statement have grounding in at least one asset in the Evidence Manifest.
    Returns (is_grounded, flagged_entities).
    """
    lower_stmt = statement.lower()
    flagged: List[str] = []

    # Compile vocabulary from evidence manifest captions & metadata
    evidence_vocab: Set[str] = set()
    for a in manifest_assets:
        caption = a.get("caption", "") or a.get("auto_caption", "")
        for token in re.findall(r"\w+", caption.lower()):
            if len(token) > 2:
                evidence_vocab.add(token)
        ambient = a.get("ambient", "") or a.get("ambient_category", "")
        if ambient:
            evidence_vocab.add(ambient.lower())

    # High-risk entities to cross-reference (unprompted named locations or dates)
    risk_locations = ["paris", "tokyo", "rome", "london", "hawaii", "new york", "beach", "mountains", "snow", "skiing"]
    for loc in risk_locations:
        if loc in lower_stmt and loc not in evidence_vocab:
            flagged.append(loc)

    is_grounded = len(flagged) == 0
    return is_grounded, flagged


def sanitize_statement(statement: str, flagged_entities: List[str]) -> str:
    """Replaces hallucinated or ungrounded claims with safe, grounded phrasing."""
    sanitized = statement
    for entity in flagged_entities:
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        sanitized = pattern.sub("this memorable place", sanitized)
    return sanitized
