from argparse import ArgumentParser, FileType, Namespace
from math import floor, sqrt
from os import listdir, path
from random import choices, seed
from statistics import mean, NormalDist, stdev
from string import punctuation
from typing import Any, Callable, Sequence

from cltk.tokenizers import LatinWordTokenizer
from tqdm import tqdm

from utils.cli.messages import BootstrapperMessage, GenericMessage
from utils.data.loaders.constants import UnitCollection, CollectionFormat
from utils.data.loaders.loader import BaseTagLoader
from utils.data.interface import get_dataset, DefinedParallelismDataset, DATASETS
from utils.data.constants import BranchRepresentation, ParallelismDirectory
from utils.data.tags import BEGINNING_TAGS, INSIDE_TAGS, BIOTag, TagLink, Tagset
from utils.stats.constants import ScoringMode, SCORING_MODES
from utils.stats.structures.match_box import MatchBox
from utils.stats.loaders import compose_bd_parallelism_directory
from utils.stats.lsa_metrics import get_lsa_score, fill_match_box, ScoreMatrix, calculate_matching_matrix_variant

# The post https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
#   was consulted for the below.


LSAMatch = tuple[int, tuple[int, int]]
StratifiedSortedDirectory = list[list[tuple[int, list[tuple[int, int]]]]]

CONJUNCTIONS: Sequence[str] = (
    "et", "at", "ac", "atque", "atqui", "autem", "uel",
    "aut", "sed", "nam", "enim", "etenim", "nec", "neque",
    "ergo", "igitur", "itaque", "tamen", "uero"
)


def pair_files(first_directory: str, second_directory: str) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    file_data_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    first_directory_items: list[str] = listdir(first_directory)
    second_directory_items: list[str] = listdir(second_directory)
    for item_index in range(0, min(len(first_directory_items), len(second_directory_items))):
        first_file_data: tuple[str, int] = (first_directory_items[item_index], (2 * item_index))
        second_file_data: tuple[str, int] = (second_directory_items[item_index], (2 * item_index) + 1)
        file_data_pair: tuple[tuple[str, int], tuple[str, int]] = (first_file_data, second_file_data)
        file_data_pairs.append(file_data_pair)

    return file_data_pairs


def gather_initial_results(first_directory_path: str, first_file_data: tuple[str, int],
                           second_directory_path: str, second_file_data: tuple[str, int],
                           loader: BaseTagLoader, loader_kwargs: dict[str, Any],
                           cleaners: list[Callable], scoring_mode: str) -> \
        tuple[MatchBox, dict[str, ParallelismDirectory], dict[str, Any]]:
    first_filename, first_file_id = first_file_data
    second_filename, second_file_id = second_file_data
    first_units: UnitCollection = loader(first_directory_path, first_filename, first_file_id, loader_kwargs)
    first_tokens, first_tags, _ = first_units[-1]
    second_units: UnitCollection = loader(second_directory_path, second_filename, second_file_id, loader_kwargs)
    second_tokens, second_tags, _ = second_units[-1]

    for cleaner in cleaners:
        cleaner(first_tokens, first_tags)
        cleaner(second_tokens, second_tags)

    if scoring_mode in (ScoringMode.EXACT_PARALLEL_MATCH, ScoringMode.MAX_PARALLEL_BRANCH_MATCH):
        branch_representation: str = BranchRepresentation.TUPLE
    else:  # scoring_mode in (ScoringMode.MAX_BRANCH_AWARE_WORD_OVERLAP, ScoringMode.MAX_WORD_OVERLAP):
        branch_representation: str = BranchRepresentation.SET

    # We gather the full parallelism directories.
    first_directory: ParallelismDirectory = compose_bd_parallelism_directory(first_tags, branch_representation)
    second_directory: ParallelismDirectory = compose_bd_parallelism_directory(second_tags, branch_representation)
    directories: dict[str, ParallelismDirectory] = {"first": first_directory, "second": second_directory}

    match_box: MatchBox = MatchBox()
    computation_steps: dict[str, Any] = calculate_matching_matrix_variant(
        first_directory, second_directory, match_box, scoring_mode, gather_steps=True
    )
    return match_box, directories, computation_steps


def gather_matched_results(sampled_matches: list[LSAMatch], score_matrix: ScoreMatrix,
                           directories: dict[str, ParallelismDirectory], scoring_mode: str) -> MatchBox:
    sample_box: MatchBox = MatchBox()

    first_directory_values, second_directory_values = \
        list(directories["first"].values()), list(directories["second"].values())
    revised_first_directory: ParallelismDirectory = {}
    revised_second_directory: ParallelismDirectory = {}

    index_rows: list[int] = []
    index_columns: list[int] = []
    indices: tuple[list[int], list[int]] = (index_rows, index_columns)
    for (row, column) in sampled_matches:
        # We build new parallelism directories. The old parallelism IDs don't particularly matter;
        # we're building these to the number of hypothesis and reference items,
        # and IDs are not used for these. Plus, it's possible that the sample contains duplicates.
        # So, storage in this manner will allow for duplicate parallelisms to exist.
        revised_first_directory[len(revised_first_directory)] = \
            first_directory_values[row] if row < len(first_directory_values) else {}
        revised_second_directory[len(revised_second_directory)] = second_directory_values[column] \
            if column < len(second_directory_values) else {}

        index_rows.append(row)
        index_columns.append(column)

    sampled_lsa_score: int = get_lsa_score(indices, score_matrix)
    sample_box.score = sampled_lsa_score
    fill_match_box(revised_first_directory, revised_second_directory, sample_box, scoring_mode)

    return sample_box


# What cases of conjunction-based issues are there, and what behavior do we want?
#   1) If every branch ...
#       a) has a conjunction either as the first token of the branch ...
#       b) or as the token before the beginning of the branch ...
#   then all conjunctions should be tagged.
#   2) If some branch doesn't start with a conjunction,
#   then no branch should start with a conjunction.


def clean_conjunctions(tokens: list[str], tags: list[list[str]]):
    tentative_directory: ParallelismDirectory = compose_bd_parallelism_directory(tags, BranchRepresentation.TUPLE)
    for parallelism_id, branches in tentative_directory.items():
        ordered_branches: list[tuple[int, int]] = list(branches)
        appropriate_stratum: int = get_stratum(ordered_branches, tags)

        conjunction_bools: list[bool] = []
        conjunction_shifted_bools: list[bool] = []
        combined_conjunction_bools: list[bool] = []
        tagged_bools: list[bool] = []
        for (branch_start, branch_end) in ordered_branches:
            conjunction_bools.append(is_conjoining(tokens[branch_start]))
            tagged_bools.append(tags[appropriate_stratum][branch_start][0] in BEGINNING_TAGS)

            if branch_start != 0:
                conjunction_shifted_bools.append(is_conjoining(tokens[branch_start - 1]))
            else:
                conjunction_shifted_bools.append(False)

            combined_conjunction_bools.append(conjunction_bools[-1] or conjunction_shifted_bools[-1])

        if all(combined_conjunction_bools) is True:
            for conjunction_index in range(0, len(conjunction_bools)):
                if conjunction_bools[conjunction_index] is True:
                    continue
                else:
                    branch_start, branch_end = ordered_branches[conjunction_index]
                    tags[appropriate_stratum][branch_start - 1] = tags[appropriate_stratum][branch_start]
                    tags[appropriate_stratum][branch_start] = BIOTag.INITIAL_INSIDE.value
        elif any(combined_conjunction_bools) is True:
            for conjunction_index in range(0, len(conjunction_bools)):
                if conjunction_bools[conjunction_index] is True:
                    branch_start, branch_end = ordered_branches[conjunction_index]
                    if (branch_start + 1) < len(tags[appropriate_stratum]):
                        tags[appropriate_stratum][branch_start + 1] = tags[appropriate_stratum][branch_start]
                        tags[appropriate_stratum][branch_start] = BIOTag.OUTSIDE.value
                    else:
                        raise ValueError("Cleaning not possible: conjunction at end of document.")
                else:
                    continue
        else:
            continue


def get_stratum(branches: list[tuple[int, int]], tags: list[list[str]]) -> int:
    stratum_number: int = 0
    for stratum in tags:
        for (branch_start, branch_end) in branches:
            if is_branch_complete(stratum, branch_start, branch_end) is False:
                break
        else:
            break

        stratum_number += 1
    else:
        raise ValueError(f"No stratum matches the branches <{branches}>.")
    return stratum_number


def is_branch_complete(stratum: list[str], branch_start: int, branch_end: int) -> bool:
    complete_bool: bool = False
    for stratum_index in range(branch_start, min(branch_end + 1, len(stratum))):
        current_tag: str = stratum[stratum_index][:1]
        if stratum_index == branch_start and current_tag not in BEGINNING_TAGS:
            break
        elif branch_start < stratum_index < branch_end and current_tag not in INSIDE_TAGS:
            break
        elif stratum_index == branch_end and current_tag in INSIDE_TAGS:
            break
    else:
        complete_bool = True
    return complete_bool


def is_conjoining(token: str) -> bool:
    conjoining_bool: bool = False
    if token in CONJUNCTIONS:
        conjoining_bool = True
    return conjoining_bool


# What are the cases of interlocking or nesting behavior, and what cleaning do we want done in response?
#   1) A parallelism exists in an A-B-A-B style structure.
#   If this is the case, then they should at most have punctuation or conjunctions between them.
#   When this happens, we want to combine branches into one parallelism.
#   2) A parallelism has segments of a parallelism nested within itself.
#   For example, B and C are two segments of A, and so it proceeds A-A and B-C-B-C.
#   In this case, we want to get rid of the Bs and Cs, leaving the As.


def clean_interlocks(tokens: list[str], tags: list[list[str]]):
    tentative_directory: ParallelismDirectory = compose_bd_parallelism_directory(tags, BranchRepresentation.TUPLE)
    stratified_parallelisms: StratifiedSortedDirectory = stratify_parallelism_directory(tentative_directory, tags)
    interlocks: list[tuple[int, set[int]]] = collect_interlocks(stratified_parallelisms, tokens)
    for (interlock_stratum, interlock) in interlocks:
        interlocked_parallelism_ids: list[int] = sorted(list(interlock))
        combine_interlocked_parallelisms(stratified_parallelisms, interlock_stratum, interlocked_parallelism_ids, tags)


def stratify_parallelism_directory(tentative_directory: ParallelismDirectory, tags: list[list[str]]) -> \
        StratifiedSortedDirectory:
    parallelisms: list[tuple[int, set[tuple[int, int]]]] = list(tentative_directory.items())
    stratified_parallelisms: StratifiedSortedDirectory = [[] for _ in tags]
    current_stratum: int = 0
    for (parallelism_id, branches) in parallelisms:
        sorted_branches: list[tuple[int, int]] = sorted(list(branches))
        updated_parallelism: tuple[int, list[tuple[int, int]]] = (parallelism_id, sorted_branches)

        if len(stratified_parallelisms[current_stratum]) > 0 and \
                sorted_branches < stratified_parallelisms[current_stratum][-1][-1]:
            current_stratum += 1
        stratified_parallelisms[current_stratum].append(updated_parallelism)

    return stratified_parallelisms


def collect_interlocks(stratified_parallelisms: StratifiedSortedDirectory, tokens: list[str]) -> \
        list[tuple[int, set[int]]]:
    interlocks: list[tuple[int, set[int]]] = []
    for stratum_number, stratum in enumerate(stratified_parallelisms):
        for parallelism_outside_index in range(0, len(stratum)):
            outside_parallelism_id, outside_parallelism = stratum[parallelism_outside_index]
            for parallelism_inside_index in range(parallelism_outside_index + 1, len(stratum)):
                inside_parallelism_id, inside_parallelism = stratum[parallelism_inside_index]
                if is_interlocking(outside_parallelism, inside_parallelism, tokens) is True:
                    new_interlock: set[int] = {outside_parallelism_id, inside_parallelism_id}
                    for (present_interlock_id, present_interlock) in interlocks:
                        if outside_parallelism_id in present_interlock or inside_parallelism_id in present_interlock:
                            present_interlock.update(new_interlock)
                            break
                    else:
                        distinct_interlock: tuple[int, set[int]] = (stratum_number, new_interlock)
                        interlocks.append(distinct_interlock)
    return interlocks


def is_interlocking(first_parallelism: list[tuple[int, int]], second_parallelism: list[tuple[int, int]],
                    tokens: list[str]) -> bool:
    interlocking_bool: bool = False
    if len(first_parallelism) == len(second_parallelism):
        sorted_first_parallelisms: list[tuple[int, int]] = sorted(list(first_parallelism))
        sorted_second_parallelisms: list[tuple[int, int]] = sorted(list(second_parallelism))
        for parallelism_index in range(0, len(sorted_first_parallelisms)):
            _, first_end = sorted_first_parallelisms[parallelism_index]
            second_start, _ = sorted_second_parallelisms[parallelism_index]
            if first_end == second_start:
                continue
            else:
                intervening_tokens: list[str] = [token for token in tokens[first_end:second_start]]
                if all((is_conjoining(token) or token in punctuation) for token in intervening_tokens):
                    continue
                else:
                    break
        else:
            interlocking_bool = True

    return interlocking_bool


def combine_interlocked_parallelisms(stratified_directory: StratifiedSortedDirectory, stratum_number: int,
                                     parallelism_ids: list[int], tags: list[list[str]]):
    interlocked_parallelisms: list[tuple[int, list[tuple[int, int]]]] = []
    for (parallelism_id, branches) in stratified_directory[stratum_number]:
        if parallelism_id in parallelism_ids:
            interlocked_parallelisms.append(branches)

    interlocked_groups: list[list[int]] = []
    for branches in interlocked_parallelisms:
        for branch_number, (branch_start, branch_end) in enumerate(branches):
            if len(interlocked_groups) <= branch_number:
                interlocked_groups.append([])

            interlocked_groups[branch_number].append(branch_start)
            interlocked_groups[branch_number].append(branch_end)

    new_branches: list[tuple[int, int]] = [(min(group), max(group)) for group in interlocked_groups]
    if stratum_number > 0 and is_parallelism_nested(stratified_directory, stratum_number, new_branches):
        for (branch_start, branch_end) in new_branches:
            for branch_index in range(branch_start, branch_end):
                tags[stratum_number][branch_index] = BIOTag.OUTSIDE.value
    else:
        for (branch_start, branch_end) in new_branches:
            for branch_index in range(branch_start, branch_end):
                if branch_index == branch_start:
                    branch_start_tag: str = tags[stratum_number][branch_index]
                    assert branch_start_tag[0] in BEGINNING_TAGS
                    if len(branch_start_tag) > 1 and int(branch_start_tag[2:]) == len(interlocked_parallelisms):
                        tags[stratum_number][branch_index] = f"{BIOTag.INITIAL_BEGINNING.value}-1"
                else:
                    tags[stratum_number][branch_index] = BIOTag.INITIAL_INSIDE.value


def is_parallelism_nested(stratified_directory: StratifiedSortedDirectory, max_stratum_number: int,
                          branches: list[tuple[int, int]]) -> bool:
    nested_bool: bool = False
    for stratum_number in range(0, max_stratum_number):
        current_stratum = stratified_directory[stratum_number]
        for (parallelism_id, ordered_branches) in current_stratum:
            if len(branches) == len(ordered_branches):
                for branch_index in range(0, len(branches)):
                    if branches[branch_index] != ordered_branches[branch_index]:
                        break
                else:
                    nested_bool = True
    return nested_bool


CLEANER_TABLE: dict[str, Callable] = {
    "conjunctions": clean_conjunctions,
    "interlocks": clean_interlocks
}

CLEANERS: Sequence[str] = tuple([cleaner for cleaner in CLEANER_TABLE.keys()])


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("first_filepath", type=str, help=BootstrapperMessage.FIRST_FILEPATH)
    parser.add_argument("second_filepath", type=str, help=BootstrapperMessage.SECOND_FILEPATH)
    parser.add_argument("--alpha", type=float, default=.05, help=BootstrapperMessage.ALPHA)
    parser.add_argument("--cleaners", type=str, nargs="*", choices=CLEANERS, help=BootstrapperMessage.CLEANERS)
    parser.add_argument(
        "--loader", type=str, choices=DATASETS, default=DefinedParallelismDataset.ASP, help=GenericMessage.LOADER
    )
    parser.add_argument(
        "--output-file", type=FileType(encoding="utf-8", mode="w+"), default="bootstrap_results.txt",
        help=BootstrapperMessage.OUTPUT_FILE
    )
    parser.add_argument("--random-seed", type=int, default=42, help=GenericMessage.RANDOM_SEED)
    parser.add_argument("--sample-count", type=int, default=1000, help=BootstrapperMessage.SAMPLE_COUNT)
    parser.add_argument("--sample-percentage", type=float, default=1.0, help=BootstrapperMessage.SAMPLE_PERCENTAGE)
    parser.add_argument(
        "--scoring-mode", choices=SCORING_MODES, default=ScoringMode.EXACT_PARALLEL_MATCH,
        help=GenericMessage.SCORING_MODE
    )
    args: Namespace = parser.parse_args()

    seed(args.random_seed)

    args.output_file.write(f"BOOTSTRAPPING RESULTS (Random Seed: {args.random_seed}):"
                           f"\n\t* Run Parameters:"
                           f"\n\t\t- Alpha: {args.alpha:.4f}"
                           f"\n\t\t- Sample Count: {args.sample_count}"
                           f"\n\t\t- Sample Percentage: {args.sample_percentage:.4f}"
                           f"\n\t\t- Scoring Mode: {args.scoring_mode}")

    # First, we pair each file together.
    if path.isdir(args.first_filepath) is True and path.isdir(args.second_filepath) is True:
        pairs: list[tuple[tuple[str, int], tuple[str, int]]] = pair_files(args.first_filepath, args.second_filepath)
    else:
        raise ValueError(f"One of <{args.first_filepath}> and <{args.second_filepath}> is not a valid directory.")

    _, dataset_loader = get_dataset(args.loader)
    loading_kwargs: dict[str, Any] = {"collection_format": CollectionFormat.DOCUMENT}
    if args.loader == DefinedParallelismDataset.ASP:
        tagging_kwargs: dict[str, Any] = {"link": TagLink.BRANCH_DISTANCE, "stratum_count": 2, "tagset": Tagset.BIO}
        loading_kwargs["tokenizer"] = LatinWordTokenizer()
    elif args.loader == DefinedParallelismDataset.PSE:
        tagging_kwargs = {"link": TagLink.TOKEN_DISTANCE, "stratum_count": 1, "tagset": Tagset.BIO}
    else:
        raise ValueError(f"The dataset <{args.loader}> is not currently recognized.")
    loading_kwargs["tagging_kwargs"] = tagging_kwargs

    cleaning_functions: list[Callable] = []
    if args.cleaners is not None:
        for cleaner in args.cleaners:
            new_function: Callable = CLEANER_TABLE[cleaner]
            if new_function not in cleaning_functions:
                cleaning_functions.append(new_function)

    # Next, we return the maximum matching produced by comparing each pair of files.
    match_boxes: list[MatchBox] = []
    parallelism_directories: list[dict[str, ParallelismDirectory]] = []
    computation_step_collection: list[dict[str, Any]] = []
    for (first_data, second_data) in tqdm(pairs, desc="Initial Computation"):
        result_match_box, result_directories, result_steps = gather_initial_results(
            args.first_filepath, first_data, args.second_filepath, second_data,
            dataset_loader, loading_kwargs, cleaning_functions, args.scoring_mode
        )

        match_boxes.append(result_match_box)
        parallelism_directories.append(result_directories)
        computation_step_collection.append(result_steps)

    args.output_file.write(f"\n\t* Initial Scores:")
    for box_index, box in enumerate(match_boxes, 0):
        args.output_file.write(f"\n\t\t- Sermon Pair <{pairs[box_index]}>: "
                               f"\n\t\t\t> Precision: {box.calculate_precision():.4f} "
                               f"({box.score} / {box.hypothesis_count})"
                               f"\n\t\t\t> Recall: {box.calculate_recall():.4f} "
                               f"({box.score} / {box.reference_count})"
                               f"\n\t\t\t> F1: {box.calculate_f_score():.4f}")

    aggregated_match_box: MatchBox = MatchBox()
    for box in match_boxes:
        aggregated_match_box += box

    args.output_file.write(f"\n\t\t- Aggregated Scores:"
                           f"\n\t\t\t> Precision: {aggregated_match_box.calculate_precision():.4f} "
                           f"({aggregated_match_box.score} / {aggregated_match_box.hypothesis_count})"
                           f"\n\t\t\t> Recall: {aggregated_match_box.calculate_recall():.4f} "
                           f"({aggregated_match_box.score} / {aggregated_match_box.reference_count})"
                           f"\n\t\t\t> F1: {aggregated_match_box.calculate_f_score():.4f}")

    # We collect all matches. We identify them based on original pair order.
    # Since sampling doesn't require any reordering of our list of pairs,
    #   using this manner of indexing is enough to recover which matches relate to which pairs.
    match_collection: list[LSAMatch] = []
    for pair_index, steps in enumerate(computation_step_collection, 0):
        rows, columns = steps["maximal_indices"]
        for i in range(0, min(len(rows), len(columns))):
            new_match: LSAMatch = (pair_index, (rows[i].item(), columns[i].item()))
            match_collection.append(new_match)

    # Following that, we perform the bootstrapping process.
    sample_size: int = floor(len(match_collection) * args.sample_percentage)
    sample_scores: list[float] = []
    args.output_file.write(f"\n\t* Sample Scores:")
    for sample in tqdm(range(0, args.sample_count), desc="Sampling"):
        # We take a sample from the matches.
        new_sample: list[LSAMatch] = choices(match_collection, k=sample_size)
        samples_by_pair: dict[int, list[LSAMatch]] = {}
        for (pair_id, sample_match) in new_sample:
            if samples_by_pair.get(pair_id, None) is None:
                samples_by_pair[pair_id] = []
            samples_by_pair[pair_id].append(sample_match)

        # For each item in the sample, we organize items back into their corresponding files and the sample score.
        sample_boxes: list[MatchBox] = []
        for (pair_id, pair_sampled_matches) in samples_by_pair.items():
            current_score_matrix: ScoreMatrix = computation_step_collection[pair_id]["score_matrix"]
            current_directories: dict[str, ParallelismDirectory] = parallelism_directories[pair_id]
            new_sample_box: MatchBox = gather_matched_results(
                pair_sampled_matches, current_score_matrix, current_directories, args.scoring_mode
            )
            sample_boxes.append(new_sample_box)

        aggregated_sample_box: MatchBox = MatchBox()
        for current_sample_box in sample_boxes:
            aggregated_sample_box += current_sample_box

        sample_score: float = aggregated_sample_box.calculate_f_score()
        sample_scores.append(sample_score)
        args.output_file.write(f"\n\t\t- Sample {sample + 1}: Aggregate Scores: "
                               f"\n\t\t\t> Precision: {aggregated_sample_box.calculate_precision():.4f} "
                               f"({aggregated_sample_box.score} / {aggregated_sample_box.hypothesis_count})"
                               f"\n\t\t\t> Recall: {aggregated_sample_box.calculate_recall():.4f} "
                               f"({aggregated_sample_box.score} / {aggregated_sample_box.reference_count})"
                               f"\n\t\t\t> F1: {aggregated_sample_box.calculate_f_score():.4f}")

    # After we take all the samples, we compute the confidence interval for them.
    sample_mean: float = mean(sample_scores)
    sample_deviation: float = stdev(sample_scores, xbar=sample_mean)
    normal_distribution: NormalDist = NormalDist()
    # We compute the appropriate z-score based on alpha.
    alpha_percentage: float = (1.0 - args.alpha)
    confidence_level: float = normal_distribution.inv_cdf((1.0 + alpha_percentage) / 2)
    confidence_lower_bound: float = sample_mean - (confidence_level * (sample_deviation / sqrt(sample_size)))
    confidence_upper_bound: float = sample_mean + (confidence_level * (sample_deviation / sqrt(sample_size)))
    confidence_interval: tuple[float, float] = (confidence_lower_bound, confidence_upper_bound)
    args.output_file.write(f"\n\t* Final Results:"
                           f"\n\t\t- Sample Mean: {sample_mean:.4f}"
                           f"\n\t\t- Sample Deviation: {sample_deviation:.4f}"
                           f"\n\t\t- Confidence Interval: [{confidence_lower_bound:.4f}, {confidence_upper_bound:.4f}] "
                           f"(Range: {(confidence_upper_bound - confidence_lower_bound):.4f})"
                           f"\n")
