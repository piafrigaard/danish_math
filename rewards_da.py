# rewards_da.py
from __future__ import annotations

from typing import List, Dict, Optional
import re

try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError as e:
    raise ImportError(
        "Please install lingua-language-detector: "
        "python3 -m pip install -U lingua-language-detector"
    ) from e


_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.DANISH, Language.ENGLISH
).build()

# strip math/latex/code so detector focuses on natural language
_LATEX_BLOCK_RE = re.compile(r"\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+(?:\{.*?\})?")  


def _clean_for_lang_detect(text: str) -> str:
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _LATEX_BLOCK_RE.sub(" ", text)
    text = _LATEX_COMMAND_RE.sub(" ", text)
    return text.strip()


def danish_minus_english_reward(
    completions: List[List[Dict[str, str]]],
    *,
    exclude_final_answer_line: bool = True,
    weight: float = 1.0,
    **kwargs,
) -> List[Optional[float]]:
    """
    Reward Danish and penalize English using Lingua confidence scores.

    For each completion:
      reward = weight * (conf_da - conf_en)

    Returns values in [-weight, +weight] (approximately; depends on Lingua).
    """
    rewards: List[Optional[float]] = []

    for completion in completions:
        content = completion[0]["content"] if completion else ""

        if exclude_final_answer_line:
            lines = content.splitlines()
            if lines:
                last = lines[-1].strip()
                if last.startswith(r"\boxed{") and last.endswith("}"):
                    content = "\n".join(lines[:-1])

        cleaned = _clean_for_lang_detect(content)

        # If there is almost no natural language, don't push the policy based on noise.
        if len(cleaned) < 15:
            rewards.append(0.0)
            continue

        confidences = _DETECTOR.compute_language_confidence_values(cleaned)

        conf_da = 0.0
        conf_en = 0.0
        for c in confidences:
            if c.language == Language.DANISH:
                conf_da = float(c.value)
            elif c.language == Language.ENGLISH:
                conf_en = float(c.value)

        r = weight * (conf_da - conf_en)
        rewards.append(r)

    return rewards
