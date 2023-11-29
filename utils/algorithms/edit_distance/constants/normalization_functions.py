from typing import Callable, Sequence


def levenshtein_normalization(source: Sequence[str], destination: Sequence[str], raw_distance: int) -> float:
    maximum_length: int = max(len(source), len(destination))
    normalized_levenshtein_distance: float = raw_distance / maximum_length
    return normalized_levenshtein_distance


NORMALIZATION_FUNCTIONS: dict[str, Callable] = {
    "levenshtein": levenshtein_normalization,
}
