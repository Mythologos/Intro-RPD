from typing import Callable, Optional, Union

from utils.data.constants import BranchRepresentation, Branch, ParallelismDirectory
from utils.stats.constants import MATCHING_SCORING_MODES, ParallelismTag, ScoringMode
from utils.data.tags import BIOTag, TagLink, BEGINNING_TAGS, FUNCTIONAL_TAGS, INSIDE_TAGS, OUTSIDE_TAGS
from utils.stats.lsa_metrics import calculate_matching_matrix_variant
from utils.stats.structures.match_box import MatchBox


def compute_tag_scoring_structure(scoring_mode: str, **kwargs) -> MatchBox:
    if kwargs.get("prediction_tags", None) is not None:
        predicted_tags: list[list[str]] = kwargs["prediction_tags"]
    else:
        raise ValueError("A list of tags is not defined for the predictions.")

    if kwargs.get("gold_tags", None) is not None:
        gold_tags: list[list[str]] = kwargs["gold_tags"]
    else:
        raise ValueError("A list of tags is not defined for the ground truth values.")

    if scoring_mode in MATCHING_SCORING_MODES:
        if scoring_mode in (ScoringMode.EXACT_PARALLEL_MATCH, ScoringMode.MAX_PARALLEL_BRANCH_MATCH):
            branch_representation: str = BranchRepresentation.TUPLE
        else:   # scoring_mode in (ScoringMode.MAX_BRANCH_AWARE_WORD_OVERLAP, ScoringMode.MAX_WORD_OVERLAP):
            branch_representation: str = BranchRepresentation.SET

        if kwargs["link"] == TagLink.TOKEN_DISTANCE:
            directory_composer: Callable = compose_td_parallelism_directory
        elif kwargs["link"] == TagLink.BRANCH_DISTANCE:
            directory_composer = compose_bd_parallelism_directory
        elif kwargs["link"] == TagLink.IDENTIFIED:
            directory_composer = compose_id_parallelism_directory
        else:
            raise ValueError(f"The link <{kwargs['link']}> is not currently supported for evaluation.")

        prediction_input: dict = directory_composer(predicted_tags, branch_representation)
        gold_input: dict = directory_composer(gold_tags, branch_representation)
        for parallelism_id, parallelism in gold_input.items():
            assert (len(parallelism) > 1)
    else:
        raise ValueError(f"The mode <{scoring_mode}> is not currently supported.")

    current_scoring_structure: MatchBox = MatchBox()
    calculate_matching_matrix_variant(prediction_input, gold_input, current_scoring_structure, scoring_mode)
    return current_scoring_structure


def get_tag_components(tag: str) -> ParallelismTag:
    if tag in OUTSIDE_TAGS or tag in FUNCTIONAL_TAGS:
        # If the model guesses an outside tag or somehow manages to guess a <START> or <STOP> tag,
        # we simply return the tag with a zero identifier.
        tag_components = ParallelismTag(tag, 0)
    else:
        tag_integer: int = int(tag[1:]) if len(tag) > 1 else 0
        tag_components = ParallelismTag(tag[0], tag_integer)
    return tag_components


def compose_id_parallelism_directory(tags: list[list[str]], branch_representation: str) -> ParallelismDirectory:
    parallelism_directory: ParallelismDirectory = {}

    for stratum in tags:
        tag_index: int = 0
        while tag_index < len(tags):
            current_tag: ParallelismTag = get_tag_components(stratum[tag_index])
            if current_tag.tag in BEGINNING_TAGS:
                branch_end_index: int = tag_index + 1
                for next_tag in stratum[tag_index + 1:]:
                    subsequent_tag: ParallelismTag = get_tag_components(next_tag)
                    if subsequent_tag.tag in INSIDE_TAGS:
                        branch_end_index += 1
                    else:
                        break

                if parallelism_directory.get(current_tag.specifier, None) is None:
                    if branch_representation == BranchRepresentation.TUPLE:
                        parallelism_directory[current_tag.specifier] = set()
                    elif branch_representation == BranchRepresentation.SET:
                        parallelism_directory[current_tag.specifier] = []
                    else:
                        raise ValueError(f"The branch representation <{branch_representation}> is not supported.")

                if branch_representation == BranchRepresentation.TUPLE:
                    branch_indices: tuple[int, int] = (tag_index, branch_end_index)
                    parallelism_directory[current_tag.specifier].add(branch_indices)
                elif branch_representation == BranchRepresentation.SET:
                    branch_token_indices: set[int] = set(range(tag_index, branch_end_index))
                    parallelism_directory[current_tag.specifier].append(branch_token_indices)   # type: ignore
                else:
                    raise ValueError(f"The branch representation <{branch_representation}> is not supported.")

                tag_index = branch_end_index
            else:
                tag_index += 1

    return parallelism_directory


def compose_td_parallelism_directory(tags: list[list[str]], branch_representation: str) -> ParallelismDirectory:
    parallelism_directory: ParallelismDirectory = {}

    for stratum in tags:
        branch_distances: dict[int, int] = {}
        tag_index: int = 0
        while tag_index < len(stratum):
            current_tag: ParallelismTag = get_tag_components(stratum[tag_index])

            if current_tag.tag == BIOTag.INITIAL_BEGINNING.value:
                branch_end_index: int = get_branch_end_index(stratum, tag_index)
                new_branch: Branch = get_new_branch(branch_representation, tag_index, branch_end_index)

                if current_tag.specifier == 0:
                    add_parallelism(new_branch, branch_representation, parallelism_directory, branch_distances)
                    linked_parallelism_id = len(parallelism_directory) - 1
                else:
                    linked_parallelism_id: Optional[int] = check_link(current_tag, branch_distances)
                    if linked_parallelism_id is None:
                        add_parallelism(new_branch, branch_representation, parallelism_directory, branch_distances)
                        linked_parallelism_id = len(parallelism_directory) - 1
                    else:
                        add_branch(branch_representation, parallelism_directory, linked_parallelism_id, new_branch)
                        branch_distances[linked_parallelism_id] = 1

                increment_value: int = branch_end_index - tag_index
                increment_distances(branch_distances, increment_value, linked_parallelism_id)
                tag_index = branch_end_index
            else:
                increment_distances(branch_distances, 1)
                tag_index += 1

    return parallelism_directory


def compose_bd_parallelism_directory(tags: list[list[str]], branch_representation: str) -> ParallelismDirectory:
    parallelism_directory: ParallelismDirectory = {}

    for stratum in tags:
        branch_distances: dict[int, int] = {}
        tag_index: int = 0
        while tag_index < len(stratum):
            current_tag: ParallelismTag = get_tag_components(stratum[tag_index])

            if current_tag.tag == BIOTag.INITIAL_BEGINNING.value:
                branch_end_index = get_branch_end_index(stratum, tag_index)
                new_branch: Branch = get_new_branch(branch_representation, tag_index, branch_end_index)

                if current_tag.specifier == 0:
                    add_parallelism(new_branch, branch_representation, parallelism_directory, branch_distances)
                    linked_parallelism_id = len(parallelism_directory) - 1
                else:
                    linked_parallelism_id: Optional[int] = check_link(current_tag, branch_distances)
                    if linked_parallelism_id is None:
                        add_parallelism(new_branch, branch_representation, parallelism_directory, branch_distances)
                        linked_parallelism_id = len(parallelism_directory) - 1
                    else:
                        add_branch(branch_representation, parallelism_directory, linked_parallelism_id, new_branch)
                        branch_distances[linked_parallelism_id] = 1

                tag_index = branch_end_index
                increment_distances(branch_distances, 1, linked_parallelism_id)
            else:
                tag_index += 1

    return parallelism_directory


def get_branch_end_index(stratum: list[str], branch_start_index: int) -> int:
    branch_end_index: int = branch_start_index + 1
    for next_tag in stratum[branch_start_index + 1:]:
        subsequent_tag: ParallelismTag = get_tag_components(next_tag)
        if subsequent_tag.tag in INSIDE_TAGS:
            branch_end_index += 1
        else:
            break

    return branch_end_index


def get_new_branch(branch_representation: str, branch_start_index: int, branch_end_index: int) -> Branch:
    if branch_representation == BranchRepresentation.TUPLE:
        new_branch: tuple[int, int] = (branch_start_index, branch_end_index)
    elif branch_representation == BranchRepresentation.SET:
        new_branch: set[int] = set(range(branch_start_index, branch_end_index))
    else:
        raise ValueError(f"The branch representation <{branch_representation}> is not supported.")

    return new_branch


def check_link(current_tag: ParallelismTag, branch_distances: dict[int, int]) -> Optional[int]:
    linked_parallelism_id: Optional[int] = None
    for parallelism_id, distance in branch_distances.items():
        if (current_tag.specifier * -1) == distance:
            linked_parallelism_id = parallelism_id
            break

    return linked_parallelism_id


def add_branch(representation: str, parallelism_directory: ParallelismDirectory, link_id: int, new_branch: Branch):
    if representation == BranchRepresentation.TUPLE:
        parallelism_directory[link_id].add(new_branch)
    elif representation == BranchRepresentation.SET:
        parallelism_directory[link_id].append(new_branch)
    else:
        raise ValueError(f"The branch representation <{representation}> is not supported.")


def add_parallelism(new_branch: Union[tuple[int, int], set[int]], branch_representation: str,
                    parallelism_directory: ParallelismDirectory, branch_distances: dict[int, int]):
    new_parallelism_id: int = len(parallelism_directory)
    if branch_representation == BranchRepresentation.TUPLE:
        parallelism_directory[new_parallelism_id] = set()
        parallelism_directory[new_parallelism_id].add(new_branch)
    elif branch_representation == BranchRepresentation.SET:
        parallelism_directory[new_parallelism_id] = [new_branch]
    else:
        raise ValueError(f"The branch representation <{branch_representation}> is not supported.")

    branch_distances[new_parallelism_id] = 1


def increment_distances(branch_distances: dict[int, int], increment_value: int, exception_id: Optional[int] = None):
    for parallelism_id in branch_distances.keys():
        if parallelism_id != exception_id:
            branch_distances[parallelism_id] += increment_value
