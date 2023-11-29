from typing import Callable, Optional

from utils.algorithms.edit_distance.constants.edits import EditOperation
from utils.algorithms.edit_distance.wf_edit_distance import calculate_minimum_edit_distance


# This serves as an example function for the cost of moves in minimum edit distance.
# In this case, current_input and proposed_output don't get used--but they could be in more complex variations.
def levenshtein_cost_function(current_input: Optional[str], proposed_output: Optional[str], move: EditOperation) -> int:
    if move == EditOperation.INSERT:
        cost = 1
    elif move == EditOperation.DELETE:
        cost = 1
    elif move == EditOperation.SUBSTITUTE:
        if current_input == proposed_output:
            cost = 0
        else:
            cost = 1
    else:
        raise ValueError(f"Invalid move <{move}> provided. Please try again with a valid move.")
    return cost


def lcs_cost_function(current_input: Optional[str], proposed_output: Optional[str], move: EditOperation) -> int:
    if move == EditOperation.INSERT:
        cost = 1
    elif move == EditOperation.DELETE:
        cost = 1
    elif move == EditOperation.SUBSTITUTE:
        if current_input == proposed_output:
            cost = 0
        else:
            cost = 2
    else:
        raise ValueError(f"Invalid move <{move}> provided. Please try again with a valid move.")
    return cost


def dual_levenshtein_cost_function(current_input: Optional[str], proposed_output: Optional[str],
                                   move: EditOperation) -> float:
    if move == EditOperation.INSERT:
        cost = 1.0
    elif move == EditOperation.DELETE:
        cost = 1.0
    elif move == EditOperation.SUBSTITUTE:
        if current_input == proposed_output:
            cost = 0.0
        else:
            edit_distance, _ = \
                calculate_minimum_edit_distance(current_input, proposed_output, COST_FUNCTIONS["levenshtein"], "int64")
            cost: float = edit_distance / max(len(current_input), len(proposed_output))
    else:
        raise ValueError(f"Invalid move <{move}> provided. Please try again with a valid move.")
    return cost


COST_FUNCTIONS: dict[str, Callable] = {
    "dual": dual_levenshtein_cost_function,
    "lcs": lcs_cost_function,
    "levenshtein": levenshtein_cost_function
}
