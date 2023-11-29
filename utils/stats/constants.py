from typing import Sequence

from aenum import NamedConstant, NamedTuple


ParallelismTag: NamedTuple = NamedTuple("ParallelismTag", "tag specifier")


class ScoringMode(NamedConstant):
    MAX_WORD_OVERLAP: str = "mwo"
    MAX_BRANCH_AWARE_WORD_OVERLAP: str = "mbawo"
    MAX_PARALLEL_BRANCH_MATCH: str = "mpbm"
    EXACT_PARALLEL_MATCH: str = "epm"


SCORING_MODES: Sequence[str] = tuple([mode for mode in ScoringMode])   # type: ignore


MATCHING_SCORING_MODES: Sequence[str] = (
    ScoringMode.EXACT_PARALLEL_MATCH, ScoringMode.MAX_PARALLEL_BRANCH_MATCH,
    ScoringMode.MAX_BRANCH_AWARE_WORD_OVERLAP, ScoringMode.MAX_WORD_OVERLAP
)
