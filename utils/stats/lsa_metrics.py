from typing import Any, Optional, Sequence

from numpy import zeros
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from utils.data.constants import ParallelismDirectory
from utils.stats.structures.match_box import MatchBox
from utils.stats.constants import ScoringMode


LSAIndices = tuple[Sequence[int], Sequence[int]]
ScoreMatrix = NDArray[int]


def calculate_matching_matrix_variant(predicted_directory: ParallelismDirectory,
                                      golden_directory: ParallelismDirectory, match_box: MatchBox,
                                      scoring_mode: str, gather_steps: bool = False) -> Optional[dict[str, Any]]:
    # We build the score matrix.
    score_matrix: NDArray[int] = get_score_matrix(predicted_directory, golden_directory, scoring_mode)
    # We use the linear sum assignment algorithm.
    maximal_indices: LSAIndices = linear_sum_assignment(score_matrix, maximize=True)   # type: ignore
    # We gather the results of the algorithm based on the maximal scores in the matrix.
    score: int = get_lsa_score(maximal_indices, score_matrix)

    # We collect the intermediate steps used to calculate the score,
    # if such a thing is requested by the user.
    if gather_steps is True:
        calculation_steps: Optional[dict[str, Any]] = {
            "score_matrix": score_matrix,
            "maximal_indices": maximal_indices,
            "score": score
        }
    else:
        calculation_steps = None

    # We fill the MatchBox data structure.
    match_box.score = score
    fill_match_box(predicted_directory, golden_directory, match_box, scoring_mode)
    return calculation_steps


def get_score_matrix(predicted_directory: ParallelismDirectory, golden_directory: ParallelismDirectory,
                     scoring_mode: str) -> NDArray[int]:
    matrix_size: int = max(len(predicted_directory.items()), len(golden_directory.items()))
    score_matrix: NDArray[int] = zeros((matrix_size, matrix_size), dtype="int64")
    for predicted_id, predicted_parallelism in enumerate(predicted_directory.values(), 0):
        for gold_id, gold_parallelism in enumerate(golden_directory.values(), 0):
            if scoring_mode == ScoringMode.EXACT_PARALLEL_MATCH:
                if predicted_parallelism == gold_parallelism:
                    score_matrix[predicted_id, gold_id] = 1
            elif scoring_mode == ScoringMode.MAX_PARALLEL_BRANCH_MATCH:
                branch_intersection = predicted_parallelism.intersection(gold_parallelism)
                score_matrix[predicted_id, gold_id] = len(branch_intersection) if len(branch_intersection) > 1 else 0
            elif scoring_mode == ScoringMode.MAX_BRANCH_AWARE_WORD_OVERLAP:
                score_matrix[predicted_id, gold_id] = get_branched_word_score(predicted_parallelism, gold_parallelism)
            elif scoring_mode == ScoringMode.MAX_WORD_OVERLAP:
                predicted_parallelism_set: set[int] = \
                    set(element for branch in predicted_parallelism for element in branch)
                gold_parallelism_set: set[int] = \
                    set(element for branch in gold_parallelism for element in branch)
                word_intersection: set[int] = predicted_parallelism_set.intersection(gold_parallelism_set)
                score_matrix[predicted_id, gold_id] = len(word_intersection)
            else:
                raise ValueError(f"The scoring mode <{scoring_mode}> is not recognized.")
    return score_matrix


def get_branched_word_score(predicted_parallelism: list[set[int]], gold_parallelism: list[set[int]]) -> int:
    matrix_dimension: int = max(len(predicted_parallelism), len(gold_parallelism))
    branched_score_matrix: ScoreMatrix = zeros((matrix_dimension, matrix_dimension), dtype="int64")
    for predicted_branch_id, predicted_branch in enumerate(predicted_parallelism, 0):
        for gold_branch_id, gold_branch in enumerate(gold_parallelism, 0):
            word_intersection: set[int] = predicted_branch.intersection(gold_branch)
            branched_score_matrix[predicted_branch_id, gold_branch_id] = len(word_intersection)

    maximal_indices: LSAIndices = linear_sum_assignment(branched_score_matrix, maximize=True)   # type: ignore
    branched_word_score: int = get_lsa_score(maximal_indices, branched_score_matrix)

    # To replicate the Heaviside function in the scoring equation corresponding to this metric,
    #   we determine the number of pairings in the match which have a nonzero contribution.
    #   If this number is more than one, then the score is valid and remains; otherwise, it is zeroed out.
    branched_word_contributors: list[int] = get_lsa_match_contributors(maximal_indices, branched_score_matrix)
    if len(branched_word_contributors) <= 1:
        branched_word_score = 0

    return branched_word_score


def get_lsa_score(indices: LSAIndices, matrix: ScoreMatrix) -> int:
    rows, columns = indices
    lsa_score: int = 0

    lsa_index: int = 0
    while lsa_index < len(rows):
        lsa_score += matrix[rows[lsa_index]][columns[lsa_index]]
        lsa_index += 1

    return lsa_score


def get_lsa_match_contributors(indices: LSAIndices, matrix: ScoreMatrix) -> list[int]:
    rows, columns = indices
    lsa_contributors: list[int] = []

    lsa_index: int = 0
    while lsa_index < len(rows):
        current_match_value: int = matrix[rows[lsa_index]][columns[lsa_index]].item()
        if current_match_value > 0:
            lsa_contributors.append(current_match_value)
        lsa_index += 1

    return lsa_contributors


# The below function assumes that there are no duplicates,
#   so it will not work if it is possible that directories can consist of duplicates (e.g., they are multisets).
#   However, since ParallelismDirectory objects are stored by ID,
#   it's possible to have a parallelism occur more than once by storing it under a distinct ID.
def fill_match_box(predicted_directory: ParallelismDirectory, golden_directory: ParallelismDirectory,
                   match_box: MatchBox, scoring_mode: str):
    if scoring_mode == ScoringMode.EXACT_PARALLEL_MATCH:
        match_box.hypothesis_count = len(predicted_directory.values())
        match_box.reference_count = len(golden_directory.values())
    elif scoring_mode == ScoringMode.MAX_PARALLEL_BRANCH_MATCH:
        match_box.hypothesis_count = sum([len(parallelism) for parallelism in predicted_directory.values()])
        match_box.reference_count = sum([len(parallelism) for parallelism in golden_directory.values()])
    elif scoring_mode in (ScoringMode.MAX_BRANCH_AWARE_WORD_OVERLAP, ScoringMode.MAX_WORD_OVERLAP):
        match_box.hypothesis_count = \
            sum([len(branch) for parallelism in predicted_directory.values() for branch in parallelism])
        match_box.reference_count = \
            sum([len(branch) for parallelism in golden_directory.values() for branch in parallelism])
    else:
        raise ValueError(f"The scoring mode <{scoring_mode}> is not recognized.")
