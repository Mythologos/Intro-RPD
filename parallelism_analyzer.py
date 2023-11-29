from argparse import ArgumentParser, BooleanOptionalAction
from itertools import chain, combinations
from math import ceil, floor
from os import listdir, path
from statistics import mean, stdev
from typing import Any, Callable, Optional, Sequence, Union

from cltk.lemmatize import LatinBackoffLemmatizer
from cltk.tokenizers import LatinWordTokenizer
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from numpy import round as np_round
from numpy.typing import NDArray
from tqdm import tqdm

from utils.algorithms.edit_distance.constants.cost_functions import COST_FUNCTIONS
from utils.algorithms.edit_distance.wf_edit_distance import calculate_minimum_edit_distance
from utils.cli.messages import GenericMessage, ParallelismMessage
from utils.data.constants import BranchRepresentation, ParallelismDirectory
from utils.data.loaders.constants import CollectionFormat, UnitCollection
from utils.data.constants import DefinedParallelismDataset, DEFINED_DATASETS
from utils.data.interface import get_dataset
from utils.data.tags import BIOTag, TagLink, Tagset
from utils.stats.loaders import compose_td_parallelism_directory

SectionedFileStatistics = dict[str, list[Union[float, list[float]]]]
AggregateFileStatistics = dict[str, float]
FileStatistics = tuple[SectionedFileStatistics, AggregateFileStatistics]
DatasetAggregateStatistics = dict[str, list[Union[float, list[float]]]]


ADAPTED_FIGURE_DIVISOR: float = 2.5
ADAPTED_FIGURE_BASE: int = 5
BASE_FONT_SIZE: int = 10
HISTOGRAM_BASE_FONT_SIZE: int = 18
FONT_SIZE_DIVISOR: int = 18
HEIGHT_CONSTANT: float = .05
MAXIMUM_DISCRETE_FREQUENCY: int = 50
MINIMUM_VALUES_FOR_ROTATION: int = 60
HISTOGRAM_BINS: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Constants:
SECTION_COUNTS: Sequence[str] = (
    "parallelism_counts", "branch_counts", "branch_sizes", "token_counts",
    "nested_parallelism_counts", "nested_branch_counts",
)

TOTAL_COUNTS: Sequence[str] = (
    "total_section_count",
    "total_parallelism_count", "total_nested_parallelism_count",
    "total_branch_count", "total_nested_branch_count",
    "total_branched_token_count", "total_token_count",
    "branches_per_parallelism",
)

COUNTS: Sequence[str] = (*SECTION_COUNTS, *TOTAL_COUNTS)

PAIRWISE_VALUES: Sequence[str] = (
    "branch_distances", 
    "absolute_size_differences", "relative_size_differences",
    "levenshtein_distances", "normalized_levenshtein_distances",
    "token_levenshtein_distances", "normalized_token_levenshtein_distances",
    "lexical_overlaps", "normalized_lexical_overlaps",
    "lemmatized_lexical_overlaps", "normalized_lemmatized_lexical_overlaps"
)

AVERAGES: Sequence[str] = (
    "average_parallelism_counts", "average_nested_parallelism_counts",
    "average_branch_counts", "average_nested_branch_counts",
    "average_branch_distances",
    "average_branch_sizes",
    "average_token_counts",
    "average_absolute_size_differences", "average_relative_size_differences",
    "average_levenshtein_distances", "average_normalized_levenshtein_distances",
    "average_token_levenshtein_distances", "average_normalized_token_levenshtein_distances",
    "average_lexical_overlaps", "average_normalized_lexical_overlaps",
    "average_lemmatized_lexical_overlaps", "average_normalized_lemmatized_lexical_overlaps",
    "average_branches_per_parallelism"
)

DEVIATIONS: Sequence[str] = (
    "parallelism_counts_deviation", "branch_counts_deviation",
    "nested_parallelism_counts_deviation", "nested_branch_counts_deviation",
    "branch_distances_deviation",
    "branch_sizes_deviation",
    "token_counts_deviation",
    "absolute_size_differences_deviation", "relative_size_differences_deviation",
    "levenshtein_distances_deviation", "normalized_levenshtein_distances_deviation",
    "token_levenshtein_distances_deviation", "normalized_token_levenshtein_distances_deviation",
    "lexical_overlaps_deviation", "normalized_lexical_overlaps_deviation",
    "lemmatized_lexical_overlaps_deviation", "normalized_lemmatized_lexical_overlaps_deviation",
    "branches_per_parallelism_deviation"
)

ALL_DATA_FIELDS: list[str] = [*COUNTS, *PAIRWISE_VALUES, *AVERAGES, *DEVIATIONS]


def expanded_mean(values: list[float]) -> float:
    if len(values) < 1:
        mean_value: int = 0
    else:
        mean_value: float = mean(values)
    return mean_value


def expanded_stdev(values: list[float]) -> float:
    if len(values) <= 1:
        standard_deviation: int = 0
    else:
        standard_deviation: float = stdev(values)
    return standard_deviation


def sorted_flatten(mapping: dict, get_sorting_criterion: Callable = lambda k_tuple: k_tuple[0],
                   get_value: Callable = lambda v_tuple: v_tuple[1]) -> list:
    flattened_mapping: list = [(mapping_key, mapping_value) for (mapping_key, mapping_value) in mapping.items()]
    flattened_mapping.sort(key=get_sorting_criterion)
    sorted_flat_data: list = [get_value(item) for item in flattened_mapping]
    return sorted_flat_data


def process_file_units(file_units: UnitCollection, **kwargs: dict[str, Any]) -> FileStatistics:
    file_results: SectionedFileStatistics = {
        "branch_counts": [],
        "absolute_size_differences": [],
        "relative_size_differences": [],
        "branch_sizes": [],
        "branch_distances": [],
        "branches_per_parallelism": [],
        "levenshtein_distances": [],
        "lemmatized_lexical_overlaps": [],
        "lexical_overlaps": [],
        "nested_branch_counts": [],
        "nested_parallelism_counts": [],
        "normalized_levenshtein_distances": [],
        "normalized_lemmatized_lexical_overlaps": [],
        "normalized_lexical_overlaps": [],
        "normalized_token_levenshtein_distances": [],
        "parallelism_counts": [],
        "token_counts": [],
        "token_levenshtein_distances": []
    }

    document_tokens: list[str] = []
    document_tags: list[list[str]] = []
    for (tokens, tags, identifiers) in file_units:
        # For a given section, we compute section-level statistics.
        compute_parallelism_count(tags, file_results)
        compute_nested_parallelism_count(tags, file_results)
        compute_branch_count(tags, file_results)
        compute_nested_branch_count(tags, file_results)
        compute_branch_sizes(tags, file_results)
        compute_branch_distances(tags, file_results)
        compute_token_count(tokens, file_results)

        # We then integrate section-level data into a document-level representation.
        document_tokens.extend(tokens)
        for stratum_index, stratum in enumerate(tags):
            if len(document_tags) == stratum_index:
                document_tags.append([])
            document_tags[stratum_index].extend(stratum)
    else:
        compute_absolute_size_differences(document_tags, file_results)
        compute_relative_size_differences(document_tags, file_results)
        compute_levenshtein_distances(document_tokens, document_tags, file_results)
        compute_levenshtein_distances(document_tokens, document_tags, file_results, should_normalize=True)
        compute_token_levenshtein_distances(document_tokens, document_tags, file_results)
        compute_token_levenshtein_distances(document_tokens, document_tags, file_results, should_normalize=True)
        compute_lexical_overlaps(document_tokens, document_tags, file_results)
        compute_lexical_overlaps(document_tokens, document_tags, file_results, should_normalize=True)
        compute_lexical_overlaps(document_tokens, document_tags, file_results, should_lemmatize=True, **kwargs)
        compute_lexical_overlaps(
            document_tokens, document_tags, file_results, should_normalize=True, should_lemmatize=True, **kwargs
        )
        compute_branches_per_parallelism(document_tags, file_results)

    file_aggregate_results: AggregateFileStatistics = compute_aggregates(file_results)

    return file_results, file_aggregate_results


def compute_parallelism_count(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_parallelism_count: int = \
        sum([tag == BIOTag.INITIAL_BEGINNING.value for stratum in tags for tag in stratum])
    file_results["parallelism_counts"].append(unit_parallelism_count)


def compute_branch_count(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_branch_count: int = \
        sum([tag.startswith(BIOTag.INITIAL_BEGINNING.value) for stratum in tags for tag in stratum])
    file_results["branch_counts"].append(unit_branch_count)


def compute_token_count(tokens: list[str], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_token_length: int = len(tokens)
    file_results["token_counts"].append(unit_token_length)


def compute_nested_parallelism_count(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_nested_parallelism_count: int = \
        sum([tag == BIOTag.INITIAL_BEGINNING.value for stratum in tags[1:] for tag in stratum])
    file_results["nested_parallelism_counts"].append(unit_nested_parallelism_count)


def compute_nested_branch_count(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_nested_branch_count: int = \
        sum([tag.startswith(BIOTag.INITIAL_BEGINNING.value) for stratum in tags[1:] for tag in stratum])
    file_results["nested_branch_counts"].append(unit_nested_branch_count)


def compute_branch_distances(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_distances: list[int] = [
        int(tag[2:]) for stratum in tags for tag in stratum
        if tag.startswith(BIOTag.INITIAL_BEGINNING.value) and len(tag) > 1
    ]
    file_results["branch_distances"].append(unit_distances)


def compute_branch_sizes(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    unit_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.SET)
    sizes: list[int] = [len(branch) for (_, branches) in unit_directory.items() for branch in branches]
    file_results["branch_sizes"].append(sizes)


def compute_absolute_size_differences(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    document_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.SET)
    absolute_size_differences: list[int] = [
        abs(len(first_branch) - len(second_branch))
        for (_, branches) in document_directory.items()
        for (first_branch, second_branch) in list(combinations(branches, 2))
    ]
    file_results["absolute_size_differences"].extend(absolute_size_differences)


def compute_relative_size_differences(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    document_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.SET)
    relative_size_differences: list[int] = [
        len(first_branch) - len(second_branch)
        for (_, branches) in document_directory.items()
        for (first_branch, second_branch) in list(combinations(branches, 2))
    ]
    file_results["relative_size_differences"].extend(relative_size_differences)


def compute_levenshtein_distances(tokens: list[str], tags: list[list[str]],
                                  file_results: dict[str, list[Union[float, list[float]]]],
                                  should_normalize: bool = False):
    document_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.TUPLE)
    levenshtein_distances: list[float] = []
    cost_function: Callable = COST_FUNCTIONS["levenshtein"]
    for (_, branches) in document_directory.items():
        for ((first_start, first_end), (second_start, second_end)) in list(combinations(branches, 2)):
            source: str = " ".join(tokens[first_start:first_end])
            destination: str = " ".join(tokens[second_start:second_end])
            edit_distance, _ = calculate_minimum_edit_distance(source, destination, cost_function, "int64")

            if should_normalize is True:
                normalization_value: int = max(len(source), len(destination))
                edit_distance = edit_distance / normalization_value

            levenshtein_distances.append(edit_distance)

    if should_normalize is False:
        file_results["levenshtein_distances"].extend(levenshtein_distances)
    else:   # should_normalize is True
        file_results["normalized_levenshtein_distances"].extend(levenshtein_distances)


def compute_token_levenshtein_distances(tokens: list[str], tags: list[list[str]],
                                        file_results: dict[str, list[Union[float, list[float]]]],
                                        should_normalize: bool = False):
    document_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.TUPLE)
    levenshtein_distances: list[float] = []
    cost_function: Callable = COST_FUNCTIONS["levenshtein"]

    for (_, branches) in document_directory.items():
        for ((first_start, first_end), (second_start, second_end)) in list(combinations(branches, 2)):
            source: list[str] = tokens[first_start:first_end]
            destination: list[str] = tokens[second_start:second_end]
            edit_distance, _ = calculate_minimum_edit_distance(source, destination, cost_function, "int64")

            if should_normalize is True:
                normalization_value: int = max(len(source), len(destination))
                edit_distance = edit_distance / normalization_value

            levenshtein_distances.append(edit_distance)

    if should_normalize is False:
        file_results["token_levenshtein_distances"].extend(levenshtein_distances)
    else:   # should_normalize is True
        file_results["normalized_token_levenshtein_distances"].extend(levenshtein_distances)


def compute_lexical_overlaps(tokens: list[str], tags: list[list[str]],
                             file_results: dict[str, list[Union[float, list[float]]]],
                             should_normalize: bool = False, should_lemmatize: bool = False,
                             lemmatizer: Optional[Union[LatinBackoffLemmatizer, Callable]] = None):
    document_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.TUPLE)
    lexical_overlaps: list[float] = []

    for (_, branches) in document_directory.items():
        for ((first_start, first_end), (second_start, second_end)) in list(combinations(branches, 2)):
            source: set[str] = create_token_multiset(tokens[first_start:first_end], should_lemmatize, lemmatizer)
            destination: set[str] = create_token_multiset(tokens[second_start:second_end], should_lemmatize, lemmatizer)
            lexical_overlap: int = len(source.intersection(destination))

            if should_normalize is True:
                normalization_value: int = len(source.union(destination))
                lexical_overlap: float = lexical_overlap / normalization_value

            lexical_overlaps.append(lexical_overlap)

    if should_normalize is False and lemmatizer is None:
        file_results["lexical_overlaps"].extend(lexical_overlaps)
    elif should_normalize is True and lemmatizer is None:
        file_results["normalized_lexical_overlaps"].extend(lexical_overlaps)
    elif should_normalize is False and lemmatizer is not None:
        file_results["lemmatized_lexical_overlaps"].extend(lexical_overlaps)
    else:   # should_normalize is True and lemmatizer is not None:
        file_results["normalized_lemmatized_lexical_overlaps"].extend(lexical_overlaps)


def create_token_multiset(tokens: list[str], should_lemmatize: bool,
                          lemmatizer: Optional[Union[LatinBackoffLemmatizer, Callable]]) -> set[str]:
    token_multiset: set[str] = set()

    if should_lemmatize is True:
        if isinstance(lemmatizer, LatinBackoffLemmatizer):
            revised_tokens = [lemma for (word, lemma) in lemmatizer.lemmatize(tokens)]
        elif isinstance(lemmatizer, Callable):
            revised_tokens = lemmatizer(tokens)
        else:
            raise ValueError(f"Unexpected lemmatizer <{lemmatizer}> with type <{type(lemmatizer)}>.")
    else:
        revised_tokens = tokens

    token_counts: dict[str, int] = {}
    for token in revised_tokens:
        if token not in token_counts:
            token_counts[token] = 0
        token_counts[token] += 1

    for (token, frequency) in token_counts.items():
        for i in range(1, frequency + 1):
            multiset_token: str = f"{token}-{i}"
            token_multiset.add(multiset_token)

    return token_multiset


def compute_branches_per_parallelism(tags: list[list[str]], file_results: dict[str, list[Union[float, list[float]]]]):
    document_directory: ParallelismDirectory = compose_td_parallelism_directory(tags, BranchRepresentation.TUPLE)
    branches_per_parallelism: list[int] = [len(branches) for parallelism_id, branches in document_directory.items()]
    file_results["branches_per_parallelism"].extend(branches_per_parallelism)


def compute_total_sections(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    aggregate_results["total_section_count"] = len(file_results["parallelism_counts"])


def compute_total_parallelisms(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    aggregate_results["total_parallelism_count"] = sum(file_results["parallelism_counts"])


def compute_total_nested_parallelisms(file_results: SectionedFileStatistics,
                                      aggregate_results: AggregateFileStatistics):
    aggregate_results["total_nested_parallelism_count"] = sum(file_results["nested_parallelism_counts"])


def compute_total_branches(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    aggregate_results["total_branch_count"] = sum(file_results["branch_counts"])


def compute_total_nested_branches(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    aggregate_results["total_nested_branch_count"] = sum(file_results["nested_branch_counts"])


def compute_total_branch_tokens(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    unit_size_totals: list[int] = [sum(unit_sizes) for unit_sizes in file_results["branch_sizes"]]
    aggregate_results["total_branched_token_count"] = sum(unit_size_totals)


def compute_total_tokens(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    aggregate_results["total_token_count"] = sum(file_results["token_counts"])


def compute_distribution_statistics(file_results: SectionedFileStatistics, aggregate_results: AggregateFileStatistics):
    for quantity_name, quantities_measured in file_results.items():
        if quantity_name in SECTION_COUNTS or quantity_name in PAIRWISE_VALUES or \
                quantity_name == "branches_per_parallelism":
            if len(quantities_measured) > 0 and isinstance(quantities_measured[0], list):
                processed_quantities = list(chain.from_iterable(quantities_measured))
            else:
                processed_quantities = quantities_measured

            aggregate_results[f"average_{quantity_name}"] = expanded_mean(processed_quantities)
            aggregate_results[f"{quantity_name}_deviation"] = expanded_stdev(processed_quantities)


TOTAL_FUNCTIONS: Sequence[Callable] = (
    compute_total_sections, compute_total_parallelisms, compute_total_nested_parallelisms, compute_total_branches,
    compute_total_nested_branches, compute_total_branch_tokens, compute_total_tokens
)


def compute_aggregates(file_results: SectionedFileStatistics) -> AggregateFileStatistics:
    aggregate_file_results: AggregateFileStatistics = {}
    for function in TOTAL_FUNCTIONS:
        function(file_results, aggregate_file_results)

    compute_distribution_statistics(file_results, aggregate_file_results)

    return aggregate_file_results


def output_file_results(filepath: str, file_results: SectionedFileStatistics,
                        file_aggregate_results: AggregateFileStatistics):
    file_branch_distances = list(chain.from_iterable(file_results['branch_distances']))
    file_branched_token_counts: list[int] = [sum(unit_sizes) for unit_sizes in file_results['branch_sizes']]
    file_branch_sizes = list(chain.from_iterable(file_results['branch_sizes']))

    normalized_edit_distance_max: Optional[float] = round(max(file_results['normalized_levenshtein_distances']), 2) \
        if file_results['normalized_levenshtein_distances'] else None
    normalized_edit_distance_min: Optional[float] = round(min(file_results['normalized_levenshtein_distances']), 2) \
        if file_results['normalized_levenshtein_distances'] else None
    token_edit_distance_max: Optional[float] = round(max(file_results['token_levenshtein_distances']), 2) \
        if file_results['token_levenshtein_distances'] else None
    token_edit_distance_min: Optional[float] = round(min(file_results['token_levenshtein_distances']), 2) \
        if file_results['token_levenshtein_distances'] else None
    normalized_token_edit_distance_max: Optional[float] = \
        round(max(file_results['normalized_token_levenshtein_distances']), 2) \
        if file_results['token_levenshtein_distances'] else None
    normalized_token_edit_distance_min: Optional[float] = \
        round(min(file_results['normalized_token_levenshtein_distances']), 2) \
        if file_results['token_levenshtein_distances'] else None
    normalized_lexical_overlap_max: Optional[float] = round(max(file_results['normalized_lexical_overlaps']), 2) \
        if file_results['normalized_lexical_overlaps'] else None
    normalized_lexical_overlap_min: Optional[float] = round(min(file_results['normalized_lexical_overlaps']), 2) \
        if file_results['normalized_lexical_overlaps'] else None
    lemmatized_lexical_overlap_max: Optional[float] = round(max(file_results['lemmatized_lexical_overlaps']), 2) \
        if file_results['lemmatized_lexical_overlaps'] else None
    lemmatized_lexical_overlap_min: Optional[float] = round(min(file_results['lemmatized_lexical_overlaps']), 2) \
        if file_results['lemmatized_lexical_overlaps'] else None
    normalized_lemmatized_lexical_overlap_max: Optional[float] = \
        round(max(file_results['normalized_lemmatized_lexical_overlaps']), 2) \
        if file_results['normalized_lemmatized_lexical_overlaps'] else None
    normalized_lemmatized_lexical_overlap_min: Optional[float] = \
        round(min(file_results['normalized_lemmatized_lexical_overlaps']), 2) \
        if file_results['normalized_lemmatized_lexical_overlaps'] else None

    with open(filepath, encoding="utf-8", mode="w+") as output_file:
        output_file.write(
            "Overall Document Statistics:"
            f"\n\t* Object Statistics:"
            f"\n\t\t** Number of Sections: {file_aggregate_results['total_section_count']};"
            f"\n\t\t** Number of Parallelisms: {file_aggregate_results['total_parallelism_count']};"
            f"\n\t\t** Number of Nested Parallelisms: {file_aggregate_results['total_nested_parallelism_count']}"
            f"\n\t\t** Number of Branches: {file_aggregate_results['total_branch_count']};"
            f"\n\t\t** Number of Nested Branches: {file_aggregate_results['total_nested_branch_count']};"
            f"\n\t\t** Number of Branched Tokens: {file_aggregate_results['total_branched_token_count']};"
            f"\n\t\t** Number of Tokens: {file_aggregate_results['total_token_count']};"
            f"\n\t* Section Statistics:"
            f"\n\t\t** Parallelisms by Section: {file_results['parallelism_counts']};"
            f"\n\t\t\t*** Average Number of Parallelisms per Section: "
            f"{file_aggregate_results['average_parallelism_counts']:.2f} \u00B1 "
            f"{file_aggregate_results['parallelism_counts_deviation']:.2f};"
            f"\n\t\t** Nested Parallelisms by Section: {file_results['nested_parallelism_counts']};"
            f"\n\t\t\t*** Average Number of Nested Parallelisms per Section: "
            f"{file_aggregate_results['average_nested_parallelism_counts']:.2f} \u00B1 "
            f"{file_aggregate_results['nested_parallelism_counts_deviation']:.2f};"
            f"\n\t\t** Branches by Section: {file_results['branch_counts']};"
            f"\n\t\t\t*** Average Number of Branches per Section: "
            f"{file_aggregate_results['average_branch_counts']:.2f} \u00B1 "
            f"{file_aggregate_results['branch_counts_deviation']:.2f};"
            f"\n\t\t** Nested Branches by Section {file_results['nested_branch_counts']};"
            f"\n\t\t\t*** Average Number of Nested Branches per Section: "
            f"{file_aggregate_results['average_nested_branch_counts']:.2f} \u00B1 "
            f"{file_aggregate_results['nested_branch_counts_deviation']:.2f};"
            f"\n\t\t** Branched Tokens by Section: {file_branched_token_counts}"
            f"\n\t\t\t*** Average Number of Branched Tokens per Section: "
            f"{expanded_mean(file_branched_token_counts):.2f} \u00B1 "
            f"{expanded_stdev(file_branched_token_counts):.2f};"
            f"\n\t\t** Tokens by Section: {file_results['token_counts']};"
            f"\n\t\t\t*** Average Number of Tokens per Section: "
            f"{file_aggregate_results['average_token_counts']:.2f} \u00B1 "
            f"{file_aggregate_results['token_counts_deviation']:.2f};"
            f"\n\t* Document Statistics:"
            f"\n\t\t** Branches per Parallelism: {file_results['branches_per_parallelism']};"
            f"\n\t\t\t*** Average Number of Branches per Parallelism: "
            f"{file_aggregate_results['average_branches_per_parallelism']:.2f} \u00B1 "
            f"{file_aggregate_results['branches_per_parallelism_deviation']:.2f};"
            f"\n\t* Distance Statistics:"
            f"\n\t\t** Branch Token Distances:"
            f"\n\t\t\t*** Average Branch Token Distance: "
            f"{file_aggregate_results['average_branch_distances']:.2f} \u00B1 "
            f"{file_aggregate_results['branch_distances_deviation']:.2f};"
            f"\n\t\t\t*** Maximum Branch Token Distance: "
            f"{max(file_branch_distances) if file_branch_distances else None};"
            f"\n\t\t\t*** Minimum Branch Token Distance: "
            f"{min(file_branch_distances) if file_branch_distances else None};"
            f"\n\t\t** Branch Sizes:"
            f"\n\t\t\t*** Average Branch Size (in Tokens): {file_aggregate_results['average_branch_sizes']:.2f} \u00B1 "
            f"{file_aggregate_results['branch_sizes_deviation']:.2f};"
            f"\n\t\t\t*** Maximum Branch Size (in Tokens): "
            f"{max(file_branch_sizes) if file_branch_sizes else None};"
            f"\n\t\t\t*** Minimum Branch Size (in Tokens): "
            f"{min(file_branch_sizes) if file_branch_sizes else None};"
            f"\n\t\t** Absolute Branch Size Differences:"
            f"\n\t\t\t*** Average Absolute Branch Size Difference: "
            f"{file_aggregate_results['average_absolute_size_differences']:.2f} \u00B1 "
            f"{file_aggregate_results['absolute_size_differences_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Absolute Branch Size Difference: "
            f"{max(file_results['absolute_size_differences']) if file_results['absolute_size_differences'] else None};"
            f"\n\t\t\t*** Minimum Absolute Branch Size Difference: "
            f"{min(file_results['absolute_size_differences']) if file_results['absolute_size_differences'] else None};"
            f"\n\t\t** Relative Branch Size Differences:"
            f"\n\t\t\t*** Average Relative Branch Size Difference: "
            f"{file_aggregate_results['average_relative_size_differences']:.2f} \u00B1 "
            f"{file_aggregate_results['relative_size_differences_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Relative Branch Size Difference: "
            f"{max(file_results['relative_size_differences']) if file_results['relative_size_differences'] else None};"
            f"\n\t\t\t*** Minimum Relative Branch Size Difference: "
            f"{min(file_results['relative_size_differences']) if file_results['relative_size_differences'] else None};"
            f"\n\t\t** Levenshtein Distances:"
            f"\n\t\t\t*** Average Levenshtein Distance: "
            f"{file_aggregate_results['average_levenshtein_distances']:.2f} \u00B1 "
            f"{file_aggregate_results['levenshtein_distances_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Levenshtein Distance: "
            f"{max(file_results['levenshtein_distances']) if file_results['levenshtein_distances'] else None};"
            f"\n\t\t\t*** Minimum Levenshtein Distance: "
            f"{min(file_results['levenshtein_distances']) if file_results['levenshtein_distances'] else None};"
            f"\n\t\t\t*** Average Normalized Levenshtein Distance: "
            f"{file_aggregate_results['average_normalized_levenshtein_distances']:.2f} \u00B1 "
            f"{file_aggregate_results['normalized_levenshtein_distances_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Normalized Levenshtein Distance: {normalized_edit_distance_max};"
            f"\n\t\t\t*** Minimum Normalized Levenshtein Distance: {normalized_edit_distance_min};"
            f"\n\t\t\t***Average Token Levenshtein Distance: "
            f"{file_aggregate_results['average_token_levenshtein_distances']:.2f} \u00B1 "
            f"{file_aggregate_results['token_levenshtein_distances_deviation']:.2f};"
            f"\n\t\t\t*** Maximum Token Levenshtein Distance: {token_edit_distance_max};"
            f"\n\t\t\t*** Minimum Token Levenshtein Distance: {token_edit_distance_min};"
            f"\n\t\t\t***Average Normalized Token Levenshtein Distance: "
            f"{file_aggregate_results['average_normalized_token_levenshtein_distances']:.2f} \u00B1 "
            f"{file_aggregate_results['normalized_token_levenshtein_distances_deviation']:.2f};"
            f"\n\t\t\t*** Maximum Normalized Token Levenshtein Distance: {normalized_token_edit_distance_max};"
            f"\n\t\t\t*** Minimum Normalized Token Levenshtein Distance: {normalized_token_edit_distance_min};"
            f"\n\t\t** Lexical Overlaps: "
            f"\n\t\t\t*** Average Lexical Overlaps: "
            f"{file_aggregate_results['average_lexical_overlaps']:.2f} \u00B1 "
            f"{file_aggregate_results['lexical_overlaps_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Lexical Overlap: "
            f"{max(file_results['lexical_overlaps']) if file_results['lexical_overlaps'] else None};"
            f"\n\t\t\t*** Minimum Lexical Overlap: "
            f"{min(file_results['lexical_overlaps']) if file_results['lexical_overlaps'] else None};"
            f"\n\t\t\t*** Average Normalized Lexical Overlaps: "
            f"{file_aggregate_results['average_normalized_lexical_overlaps']:.2f} \u00B1 "
            f"{file_aggregate_results['normalized_lexical_overlaps_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Normalized Lexical Overlaps: {normalized_lexical_overlap_max};"
            f"\n\t\t\t*** Minimum Normalized Lexical Overlaps: {normalized_lexical_overlap_min};"
            f"\n\t\t\t*** Average Lemmatized Lexical Overlaps: "
            f"{file_aggregate_results['average_lemmatized_lexical_overlaps']:.2f} \u00B1 "
            f"{file_aggregate_results['lemmatized_lexical_overlaps_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Normalized Lexical Overlaps: {lemmatized_lexical_overlap_max};"
            f"\n\t\t\t*** Minimum Normalized Lexical Overlaps: {lemmatized_lexical_overlap_min};"
            f"\n\t\t\t*** Average Normalized Lemmatized Lexical Overlaps: "
            f"{file_aggregate_results['average_normalized_lemmatized_lexical_overlaps']:.2f} \u00B1 "
            f"{file_aggregate_results['normalized_lemmatized_lexical_overlaps_deviation']:.2f}"
            f"\n\t\t\t*** Maximum Normalized Lemmatized Lexical Overlaps: {normalized_lemmatized_lexical_overlap_max};"
            f"\n\t\t\t*** Minimum Normalized Lemmatized Lexical Overlaps: {normalized_lemmatized_lexical_overlap_min};"
        )


def output_aggregated_results(filepath: str, dataset_results: DatasetAggregateStatistics):
    empty_lexical_overlaps: list[int] = [1 if overlap == 0 else 0 for overlap in dataset_results['lexical_overlaps']]
    total_lexical_overlaps: list[int] = \
        [1 if overlap == 1.0 else 0 for overlap in dataset_results['normalized_lexical_overlaps']]

    with open(filepath, encoding="utf-8", mode="w+") as output_file:
        output_file.write(
            f"Dataset Statistics (Pre-Aggregation):"
            f"\n\t* Average Number of Parallelisms per Section: "
            f"{expanded_mean(dataset_results['parallelism_counts']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['parallelism_counts']):.2f}"
            f"\n\t* Average Number of Nested Parallelisms per Section: "
            f"{expanded_mean(dataset_results['nested_parallelism_counts']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['nested_parallelism_counts']):.2f}"
            f"\n\t* Average Number of Branches per Section: {expanded_mean(dataset_results['branch_counts']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['branch_counts']):.2f}"
            f"\n\t* Average Number of Nested Branches per Section: "
            f"{expanded_mean(dataset_results['nested_branch_counts']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['nested_branch_counts']):.2f}"
            f"\n\t* Average Number of Branches per Parallelism: "
            f"{expanded_mean(dataset_results['branches_per_parallelism']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['branches_per_parallelism']):.2f}"            
            f"\n\t* Average Branch Distance: {expanded_mean(dataset_results['branch_distances']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['branch_distances']):.2f}"
            f"\n\t* Average Branch Size: {expanded_mean(dataset_results['branch_sizes']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['branch_sizes']):.2f}"
            f"\n\t* Average Absolute Size Difference: "
            f"{expanded_mean(dataset_results['absolute_size_differences']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['absolute_size_differences']):.2f}"
            f"\n\t* Average Relative Size Difference: "
            f"{expanded_mean(dataset_results['relative_size_differences']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['relative_size_differences']):.2f}"
            f"\n\t* Average Levenshtein Distance: "
            f"{expanded_mean(dataset_results['levenshtein_distances']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['levenshtein_distances']):.2f} "
            f"\n\t* Average Normalized Levenshtein Distance: "
            f"{expanded_mean(dataset_results['normalized_levenshtein_distances']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['normalized_levenshtein_distances']):.2f} "
            f"\n\t* Average Token Levenshtein Distance: "
            f"{expanded_mean(dataset_results['token_levenshtein_distances']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['token_levenshtein_distances']):.2f} "
            f"\n\t* Average Normalized Token Levenshtein Distance: "
            f"{expanded_mean(dataset_results['normalized_token_levenshtein_distances']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['normalized_token_levenshtein_distances']):.2f} "
            f"\n\t* Average Lexical Overlap: "
            f"{expanded_mean(dataset_results['lexical_overlaps']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['lexical_overlaps']):.2f} "
            f"\n\t* Percentage of Pairs with No Lexical Overlap: "
            f"{(sum(empty_lexical_overlaps) / len(empty_lexical_overlaps)) * 100:.2f}%"
            f"\n\t* Percentage of Pairs with Full Lexical Overlap: "
            f"{(sum(total_lexical_overlaps) / len(total_lexical_overlaps)) * 100:.2f}%"
            f"\n\t* Average Normalized Lexical Overlap: "
            f"{expanded_mean(dataset_results['normalized_lexical_overlaps']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['normalized_lexical_overlaps']):.2f} "
            f"\n\t* Average Lemmatized Lexical Overlap: "
            f"{expanded_mean(dataset_results['lemmatized_lexical_overlaps']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['lemmatized_lexical_overlaps']):.2f} "
            f"\n\t* Average Normalized Lemmatized Lexical Overlap: "
            f"{expanded_mean(dataset_results['normalized_lemmatized_lexical_overlaps']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['normalized_lemmatized_lexical_overlaps']):.2f} "

            f"\n\n"
            f"Dataset Statistics (Post-Aggregation):"
            f"\n\t* Average Section Count per Document: {expanded_mean(dataset_results['total_section_count']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['total_section_count']):.2f}"
            f"\n\t* Average Number of Parallelisms per Document: "
            f"{expanded_mean(dataset_results['total_parallelism_count']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['total_parallelism_count']):.2f}"
            f"\n\t* Average Number of Branches per Section: {expanded_mean(dataset_results['total_branch_count']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['total_branch_count']):.2f}"
            f"\n\t* Average Number of Branched Tokens per Section: "
            f"{expanded_mean(dataset_results['total_branched_token_count']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['total_branched_token_count']):.2f}"
            f"\n\t* Average Number of Tokens per Section: "
            f"{expanded_mean(dataset_results['total_token_count']):.2f} "
            f"\u00B1 {expanded_stdev(dataset_results['total_token_count']):.2f}"
            f"\n\t* Total Number of Tokens: {sum(dataset_results['total_token_count'])}"
            f"\n\t* Total Number of Branched Tokens: {sum(dataset_results['total_branched_token_count'])}"
            f"\n\t* Total Number of Branches: {sum(dataset_results['total_branch_count'])}"
            f"\n\t* Total Number of Nested Branches: {sum(dataset_results['total_nested_branch_count'])}"
            f"\n\t* Total Number of Parallelisms: {sum(dataset_results['total_parallelism_count'])}"
            f"\n\t* Total Number of Nested Parallelisms: {sum(dataset_results['total_nested_parallelism_count'])}"
            f"\n\t* Total Number of Sections: {sum(dataset_results['total_section_count'])}"
        )


def visualize_integral_frequency_distribution(data: list[int], x_label: str, y_label: str, plot_title: str,
                                              should_bucket: bool = False):
    figure_size: tuple[int, int] = get_adapted_data_size(data)
    fig, ax = plt.subplots(figsize=figure_size)

    # First, we tally frequencies of the data within the given range, which should be the entire range of the data.
    value_minimum: int = min(0, min(data))
    value_maximum: int = max(1, max(data) + 1)
    values: dict[int, int] = get_data_by_frequency(data, value_minimum, value_maximum)

    # Next, we check to see whether we want to "bucket" any empty terms, since many relevant distributions have them.
    if should_bucket is True:
        buckets: list[list[int]] = [[]]
        for (label, frequency) in values.items():
            if frequency == 0:
                buckets[-1].append(label)   # We continue filling a given bucket.
            else:
                if len(buckets[-1]) != 0:
                    buckets.append([])   # We add another empty list, as this cues a new bucket to be formed.
                else:
                    continue   # We do nothing, as we are still waiting to fill the currently-empty bucket.

        for bucket in buckets:
            if len(bucket) > 2:  # Buckets with two items aren't abbreviated.
                bucket.sort()
                for index in bucket[1:-1]:
                    del values[index]
                    # The order of values will be represented by the smallest and largest empty indices.

    if should_bucket is False:
        x_coordinates: list[int] = [range_value for range_value in range(value_minimum, value_maximum)]
        bar_frequencies: list[int] = sorted_flatten(values)
        tick_values: list[int] = [range_value for range_value in range(value_minimum, value_maximum)]
    else:  # should_bucket is True
        x_coordinates: list[int] = sorted([value_key for value_key in values.keys()])
        bar_frequencies: list[int] = []
        tick_values: list[str] = []

        current_index: int = 0
        while current_index < len(x_coordinates):
            current_coordinate = x_coordinates[current_index]
            subsequent_index: int = current_index + 1
            # If the next value isn't numerically adjacent, then we add a range-based label.
            # We delete the high end of the range, since we use the low end to "represent" it.
            if subsequent_index < len(x_coordinates) and current_coordinate + 1 != x_coordinates[subsequent_index]:
                tick_values.append(f"{current_coordinate}\u2013{x_coordinates[subsequent_index]}")
                del x_coordinates[subsequent_index]
            else:
                tick_values.append(str(current_coordinate))

            bar_frequencies.append(values[current_coordinate])
            current_index += 1

        # We re-evaluate the x-coordinates to make presentation more concise.
        x_coordinates = [range_value for range_value in range(value_minimum, value_minimum + len(bar_frequencies))]

    font_size: int = BASE_FONT_SIZE - floor(len(x_coordinates) / FONT_SIZE_DIVISOR)
    vertical_bars = ax.bar(x_coordinates, bar_frequencies, color=get_colors(len(bar_frequencies), 'magma'))
    ax.bar_label(vertical_bars, fontsize=font_size)

    if len(x_coordinates) <= MINIMUM_VALUES_FOR_ROTATION:
        rotation_value: int = 0
    else:
        rotation_value: int = 75

    ax.set_xticks(x_coordinates, labels=tick_values, rotation=rotation_value, fontsize=font_size)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)

    plt.show()


def visualize_rational_histogram(data: list[float], x_label: str, y_label: str, plot_title: str):
    bins: list[float] = HISTOGRAM_BINS
    adapted_figure_dim_size: int = ADAPTED_FIGURE_BASE + len(HISTOGRAM_BINS) // 2
    figure_size: tuple[int, int] = (adapted_figure_dim_size, adapted_figure_dim_size)
    fig, ax = plt.subplots(figsize=figure_size)
    _, histogram_bins, patches = ax.hist(data, bins, color="cornflowerblue", edgecolor="midnightblue", linewidth=2)

    data_values: NDArray[float] = patches.datavalues
    percentages: list[str] = []
    for index in range(0, len(data_values)):
        base_percentage: float = np_round((data_values[index] / data_values.sum()) * 100, decimals=1).item()
        if 1.0 > base_percentage > 0.0 and data_values[index] != 0.0:
            percentage_string: str = "<1%"
        else:
            percentage_string: str = f"{str(round(base_percentage))}%"
        percentages.append(percentage_string)

    y_ticks: list[float] = ax.get_yticks()
    y_ticks: list[int] = [int(tick) for tick in y_ticks]
    ax.bar_label(patches, labels=percentages, fontsize=HISTOGRAM_BASE_FONT_SIZE)
    ax.set_xticks(HISTOGRAM_BINS, fontsize=HISTOGRAM_BASE_FONT_SIZE, labels=HISTOGRAM_BINS)
    ax.set_yticks(y_ticks, fontsize=HISTOGRAM_BASE_FONT_SIZE, labels=y_ticks)
    ax.set_xlabel(x_label, fontsize=HISTOGRAM_BASE_FONT_SIZE + 2)
    ax.set_ylabel(y_label, fontsize=HISTOGRAM_BASE_FONT_SIZE + 2)
    ax.set_title(plot_title, fontsize=HISTOGRAM_BASE_FONT_SIZE + 4)
    plt.tight_layout()
    plt.show()


def get_data_by_frequency(data: list[int], minimum_value: int, maximum_value: int) -> dict[int, int]:
    data_by_frequency: dict[int, int] = {range_value: 0 for range_value in range(minimum_value, maximum_value)}
    for data_point in data:
        data_by_frequency[data_point] += 1
    return data_by_frequency


def get_colors(num_colors: int, map_name: str):
    color_map: Colormap = cm.get_cmap(map_name)
    current_color_value: float = 0.0
    incrementation_value: float = 1.0 / num_colors

    colors: list[tuple[float, float, float, float]] = []
    color_counter: int = 0
    while color_counter < num_colors:
        colors.append(color_map(current_color_value))
        color_counter += 1
        current_color_value += incrementation_value
    return colors


def get_adapted_data_size(data: list, bin_size: int = 1) -> tuple[int, int]:
    unique_value_count: int = len(set(data))
    if bin_size > 1:
        unique_value_count = ceil(unique_value_count / 2)
    adapted_figure_size: int = ceil(unique_value_count / ADAPTED_FIGURE_DIVISOR) + ADAPTED_FIGURE_BASE
    figure_size: tuple[int, int] = (adapted_figure_size, adapted_figure_size)
    return figure_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_directory", type=str, help=ParallelismMessage.INPUT_DIRECTORY)
    parser.add_argument("output_directory", type=str, help=ParallelismMessage.OUTPUT_DIRECTORY)
    parser.add_argument("--aggregate", action=BooleanOptionalAction, default=False, help=ParallelismMessage.AGGREGATE)
    parser.add_argument(
        "--loader", type=str, choices=DEFINED_DATASETS, default=DefinedParallelismDataset.ASP,
        help=GenericMessage.LOADER
    )
    parser.add_argument("--visualize", action=BooleanOptionalAction, default=False, help=ParallelismMessage.VISUALIZE)
    args = parser.parse_args()

    if args.aggregate is True:
        full_results: Optional[dict[str, list[float]]] = {}
        for data_field in ALL_DATA_FIELDS:
            full_results[data_field] = []
    else:
        full_results = None

    _, dataset_loader = get_dataset(args.loader)
    processing_kwargs: dict[str, Any] = {}
    if args.loader == DefinedParallelismDataset.ASP:
        loading_kwargs: dict[str, Any] = {
            "collection_format": CollectionFormat.SECTION,
            "tagging_kwargs": {"link": TagLink.TOKEN_DISTANCE, "stratum_count": 2, "tagset": Tagset.BIO},
            "tokenizer": LatinWordTokenizer()
        }
        processing_kwargs["lemmatizer"] = LatinBackoffLemmatizer()
    elif args.loader == DefinedParallelismDataset.PSE:
        loading_kwargs: dict[str, Any] = {
            "collection_format": CollectionFormat.SECTION,
            "tagging_kwargs": {"link": TagLink.TOKEN_DISTANCE, "stratum_count": 1, "tagset": Tagset.BIO},
        }
        processing_kwargs["lemmatizer"] = lambda x: x
    else:
        raise ValueError(f"The dataset <{args.loader}> is not currently recognized.")

    formatting_kwargs: dict[str, Any] = {}

    if path.isdir(args.input_directory) and path.isdir(args.output_directory):
        input_filenames: list[str] = [input_filename for input_filename in listdir(args.input_directory)]
        for input_filename in tqdm(input_filenames):
            name, extension = input_filename.rsplit(".", maxsplit=1)

            # First, we load the file.
            current_units: UnitCollection = dataset_loader(args.input_directory, input_filename, name, loading_kwargs)

            # Then, we process the file.
            current_results, current_aggregate_results = process_file_units(current_units, **processing_kwargs)

            # Finally, we output the file.
            output_filepath: str = f"{args.output_directory}/{name}_stats.txt"
            output_file_results(output_filepath, current_results, current_aggregate_results)

            if args.aggregate is True and full_results is not None:
                for (result_name, results_list) in current_results.items():
                    if result_name in ("branch_distances", "branch_sizes"):
                        revised_results = list(chain.from_iterable(results_list))
                    else:
                        revised_results = results_list

                    full_results[result_name].extend(revised_results)

                for (result_name, result) in current_aggregate_results.items():
                    full_results[result_name].append(result)

        if args.aggregate is True:
            aggregation_filename: str = f"{args.output_directory}/aggregate_stats.txt"
            output_aggregated_results(aggregation_filename, full_results)

            if args.visualize is True:
                for (statistic, results) in full_results.items():
                    title: str = statistic.replace("_", " ").title()
                    if statistic in ("normalized_lexical_overlaps", "normalized_lemmatized_lexical_overlaps"):
                        print(f"Displaying <{title}>...")
                        visualize_rational_histogram(results, "Percentile Bins", "Frequency", title)
                    elif isinstance(results, list) and all([isinstance(result, int) for result in results]):
                        print(f"Displaying <{title}>...")
                        visualize_integral_frequency_distribution(results, "Values", "Frequencies", title, True)
    elif path.isdir(args.input_filepath) and not path.isdir(args.output_filepath):
        raise ValueError("An invalid output directory was given. This directory must exist.")
    else:
        raise ValueError(f"The input filepath given is not a valid file or directory.")
