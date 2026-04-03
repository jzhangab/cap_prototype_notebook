"""
LLM-based intent classifier with chain-of-thought reasoning.

Flow:
  1. Ask the LLM to reason step-by-step about what the user needs.
  2. Extract the structured JSON from that reasoning response.
  3. Apply a confidence threshold; fall back to regex extraction if JSON fails.
"""
from __future__ import annotations

import json
import logging
import re

from backend.llm.llm_client import LLMClient
from backend.llm.prompt_templates import (INTENT_CLASSIFIER_SYSTEM,
                                           INTENT_CLASSIFIER_USER)

logger = logging.getLogger(__name__)

VALID_INTENTS = {
    "site_list_merger",
    "trial_benchmarking",
    "drug_reimbursement",
    "enrollment_forecasting",
}

# Lower threshold: the reasoning step makes the model more deliberate,
# so we can accept matches it is moderately confident in.
CONFIDENCE_THRESHOLD = 0.55


def classify_intent(
    llm: LLMClient,
    user_message: str,
    history: list[dict],
) -> tuple[str | None, float, str]:
    """
    Returns (intent_or_None, confidence, reasoning).
    Returns (None, ...) when intent is unknown or below the confidence threshold.
    """
    history_text = _format_history(history)

    messages = [
        {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM},
        {"role": "user", "content": INTENT_CLASSIFIER_USER.format(
            history=history_text,
            user_message=user_message,
        )},
    ]

    try:
        raw = llm.complete(messages, temperature=llm.temp_classify)
    except Exception as e:
        logger.error("Intent classification LLM call failed: %s", e)
        return None, 0.0, "LLM call failed"

    intent, confidence, reasoning = _parse_response(raw)

    logger.info(
        "Intent classification — intent=%s confidence=%.2f reasoning=%s",
        intent, confidence, reasoning
    )

    if intent not in VALID_INTENTS or confidence < CONFIDENCE_THRESHOLD:
        return None, confidence, reasoning

    return intent, confidence, reasoning


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> tuple[str, float, str]:
    """
    Try to extract (intent, confidence, reasoning) from the LLM response.
    Attempts full JSON parse first; falls back to regex extraction so that
    minor formatting deviations from the model don't break classification.
    """
    # --- Attempt 1: clean JSON parse ---
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = "\n".join(
                l for l in text.splitlines() if not l.strip().startswith("```")
            ).strip()
        data = json.loads(text)
        intent    = str(data.get("intent", "unknown")).strip().lower()
        confidence = float(data.get("confidence", 0.0))
        reasoning  = str(data.get("reasoning", ""))
        return intent, confidence, reasoning
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # --- Attempt 2: regex field extraction from free-form text ---
    intent     = _regex_extract_intent(raw)
    confidence = _regex_extract_confidence(raw)
    reasoning  = _regex_extract_reasoning(raw)
    if intent:
        logger.warning(
            "JSON parse failed; recovered via regex — intent=%s confidence=%.2f",
            intent, confidence
        )
        return intent, confidence, reasoning

    logger.error("Could not parse intent from LLM response: %s", raw[:300])
    return "unknown", 0.0, raw[:200]


def _regex_extract_intent(text: str) -> str:
    """Look for a valid skill_id anywhere in the response text."""
    for skill_id in VALID_INTENTS:
        if re.search(rf'\b{re.escape(skill_id)}\b', text, re.IGNORECASE):
            return skill_id
    # Also match quoted partial forms
    patterns = {
        "site_list_merger":       r'\b(site.?list.?merg|merg.*site)\b',
        "trial_benchmarking":     r'\b(trial.?benchmark|benchmark.?trial)\b',
        "drug_reimbursement":     r'\b(reimbursement|drug.?reimburse)\b',
        "enrollment_forecasting": r'\b(enrollment.?forecast|forecast.?enroll)\b',
    }
    for skill_id, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return skill_id
    return "unknown"


def _regex_extract_confidence(text: str) -> float:
    """Extract the first float that looks like a confidence score."""
    # Try "confidence": 0.85 pattern first
    m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m:
        try:
            return min(1.0, max(0.0, float(m.group(1))))
        except ValueError:
            pass
    # Fallback: any standalone decimal in [0,1]
    for token in re.findall(r'\b0\.\d+\b', text):
        try:
            v = float(token)
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return 0.6   # Default moderate confidence if we found a skill via regex


def _regex_extract_reasoning(text: str) -> str:
    """Extract reasoning field or return the first two sentences."""
    m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(sentences[:2]) if sentences else text[:200]


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    return "\n".join(
        f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
        for m in history
    )
