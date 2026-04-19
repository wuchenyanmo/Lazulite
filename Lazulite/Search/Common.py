from __future__ import annotations

from fuzzywuzzy import fuzz


def combined_fuzzy_score(str1: str, str2: str, full_match_weight: float = 0.2) -> float:
    partial = fuzz.partial_ratio(str1, str2)
    full = fuzz.ratio(str1, str2)
    return float(partial * (1 - full_match_weight) + full * full_match_weight)
