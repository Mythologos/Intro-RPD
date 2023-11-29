from argparse import ArgumentParser, Namespace
from copy import deepcopy
from math import inf
from os import listdir, mkdir, path
from shutil import copy
from statistics import mean, stdev
from typing import Any

from cltk.tokenizers.lat.lat import LatinWordTokenizer
from numpy import argmin
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from utils.cli.constants import DEFAULT_SPLITS
from utils.cli.messages import SplitMessage, GenericMessage
from utils.data.constants import DEFINED_DATASETS
from utils.data.loaders.constants import TagUnit, UnitCollection, CollectionFormat
from utils.data.interface import get_dataset, DefinedParallelismDataset
from utils.data.loaders.loader import BaseTagLoader
from utils.data.tags import TagLink, Tagset, OUTSIDE_TAGS


def count_nested_tags(input_path: str, input_filename: str, backup_id: int,
                      loader: BaseTagLoader, loader_kwargs: dict[str, Any]) -> tuple[int, int]:
    outside_tags: int = 0
    inside_tags: int = 0

    units: UnitCollection = loader(input_path, input_filename, backup_id, loader_kwargs)
    document_unit: TagUnit = units[0]
    _, tags, _ = document_unit
    for stratum in tags:
        for tag in stratum:
            if tag in OUTSIDE_TAGS:
                outside_tags += 1
            else:
                inside_tags += 1

    return outside_tags, inside_tags


def split_by_tag_mse(triples: list[tuple[int, int, str]], sets: list[str], ratios: list[float]) -> dict[str, list[str]]:
    current_inside_values: list[int] = [0 for _ in range(0, len(sets))]
    current_outside_values: list[int] = [0 for _ in range(0, len(sets))]

    total_in_values: int = 0
    total_out_values: int = 0

    set_divisions: dict[str, list[str]] = {}
    set_inside_counts: dict[str, list[int]] = {}
    set_outside_counts: dict[str, list[int]] = {}
    for (new_inside, new_outside, new_filename) in tqdm(triples, desc="MSE Sorting"):
        mse_values: list[float] = [inf for _ in range(0, len(ratios))]
        total_in_values += new_inside
        total_out_values += new_outside
        for set_index in range(0, len(sets)):
            prospective_inside_values: list[float] = deepcopy(current_inside_values)
            prospective_outside_values: list[float] = deepcopy(current_outside_values)
            prospective_inside_values[set_index] += new_inside
            prospective_outside_values[set_index] += new_outside
            prospective_inside_values = [value / total_in_values for value in prospective_inside_values]
            prospective_outside_values = [value / total_out_values for value in prospective_outside_values]

            inside_mse: float = mean_squared_error(ratios, prospective_inside_values)
            outside_mse: float = mean_squared_error(ratios, prospective_outside_values)
            mse_values[set_index] = (inside_mse + outside_mse) / 2
        else:
            selected_index: int = argmin(mse_values).item()
            current_inside_values[selected_index] += new_inside
            current_outside_values[selected_index] += new_outside
            if set_divisions.get(args.sets[selected_index], None) is None:
                set_divisions[args.sets[selected_index]] = []
                set_inside_counts[args.sets[selected_index]] = []
                set_outside_counts[args.sets[selected_index]] = []
            set_divisions[args.sets[selected_index]].append(new_filename)
            set_inside_counts[args.sets[selected_index]].append(new_inside)
            set_outside_counts[args.sets[selected_index]].append(new_outside)

    display_kwargs: dict[str, Any] = {
        "total_in_values": total_in_values,
        "total_out_values": total_out_values,
        "set_inside_values": current_inside_values,
        "set_outside_values": current_outside_values,
        "set_inside_counts": set_inside_counts,
        "set_outside_counts": set_outside_counts,
        "set_divisions": set_divisions
    }

    display_split_results(sets, ratios, **display_kwargs)

    return set_divisions


def split_by_matching_directory(triples: list[tuple[int, int, str]], sets: list[str], ratios: list[float],
                                match_directory: str) -> dict[str, list[str]]:
    alphabetical_set_divisions: dict[str, list[str]] = {set_name: [] for set_name in sets}
    set_inside_counts: dict[str, list[int]] = {set_name: [] for set_name in sets}
    set_outside_counts: dict[str, list[int]] = {set_name: [] for set_name in sets}
    for set_name in sets:
        set_filenames: list[str] = listdir(f"{match_directory}/{set_name}")
        for set_filename in set_filenames:
            alphabetical_set_divisions[set_name].append(set_filename)

    sorted_set_divisions: dict[str, list[str]] = {set_name: [] for set_name in sets}
    total_in_values: int = 0
    total_out_values: int = 0
    set_inside_values: list[int] = [0 for _ in range(0, len(sets))]
    set_outside_values: list[int] = [0 for _ in range(0, len(sets))]
    for (inside_tag_count, outside_tag_count, current_filename) in triples:
        total_in_values += inside_tag_count
        total_out_values += outside_tag_count
        for (division_index, (division, filenames)) in enumerate(alphabetical_set_divisions.items(), start=0):
            if current_filename in filenames:
                sorted_set_divisions[division].append(current_filename)
                set_inside_values[division_index] += inside_tag_count
                set_outside_values[division_index] += outside_tag_count
                set_inside_counts[division].append(inside_tag_count)
                set_outside_counts[division].append(outside_tag_count)
                break

    display_kwargs: dict[str, Any] = {
        "total_in_values": total_in_values,
        "total_out_values": total_out_values,
        "set_inside_values": set_inside_values,
        "set_outside_values": set_outside_values,
        "set_inside_counts": set_inside_counts,
        "set_outside_counts": set_outside_counts,
        "set_divisions": sorted_set_divisions
    }
    display_split_results(sets, ratios, **display_kwargs)

    return sorted_set_divisions


def display_split_results(sets: list[str], ratios: list[float], **kwargs):
    total_in_values: int = kwargs["total_in_values"]
    total_out_values: int = kwargs["total_out_values"]
    set_inside_values: list[int] = kwargs["set_inside_values"]
    set_outside_values: list[int] = kwargs["set_outside_values"]
    set_inside_counts: dict[str, list[int]] = kwargs["set_inside_counts"]
    set_outside_counts: dict[str, list[int]] = kwargs["set_outside_counts"]
    set_divisions: dict[str, list[str]] = kwargs["set_divisions"]

    final_inside_ratios: list[float] = [value / total_in_values for value in set_inside_values]
    final_outside_ratios: list[float] = [value / total_in_values for value in set_outside_values]
    final_inside_error: float = mean_squared_error(ratios, final_inside_ratios)
    final_outside_error: float = mean_squared_error(ratios, final_outside_ratios)
    final_error: float = (final_inside_error + final_outside_error) / 2

    print(f"The overall error in dividing up the files is: {final_error:.4f}.")
    print("The division of files is as follows: ")
    for set_index in range(0, len(sets)):
        tag_ratios: list[float] = [
            set_inside_counts[sets[set_index]][count_index] / set_outside_counts[sets[set_index]][count_index]
            for count_index in range(0, len(set_inside_counts[sets[set_index]]))
        ]
        print(f"{sets[set_index]}: {len(set_divisions[sets[set_index]])} Files")
        print(f"\t* Inside Tags: {set_inside_values[set_index]} "
              f"({set_inside_values[set_index] / total_in_values:.4f})")
        print(f"\t* Outside Tags: {set_outside_values[set_index]} "
              f"({set_outside_values[set_index] / total_out_values:.4f})")
        print(f"\t* Overall Statistics:"
              f"\n\t\t** Average Inside Tags: {mean(set_inside_counts[sets[set_index]]):.4f} \u00b1 "
              f"{stdev(set_inside_counts[sets[set_index]]):.4f}"
              f"\n\t\t** Average Outside Tags: {mean(set_outside_counts[sets[set_index]]):.4f} \u00b1 "
              f"{stdev(set_outside_counts[sets[set_index]]):.4f}"
              f"\n\t\t** Average Tag Ratio: {mean(tag_ratios):.4f} \u00b1 {stdev(tag_ratios):.4f} ")

        for other_set_index in range(set_index + 1, len(sets)):
            inside_statistic, inside_p_value = ttest_ind(
                set_inside_counts[sets[set_index]], set_inside_counts[sets[other_set_index]], equal_var=False
            )
            outside_statistic, outside_p_value = ttest_ind(
                set_outside_counts[sets[set_index]], set_outside_counts[sets[other_set_index]], equal_var=False
            )
            print(f"\t\t** Welch's T-Test Results ({sets[set_index]} vs. {sets[other_set_index]}):"
                  f"\n\t\t\t*** Inside: Statistic - {inside_statistic:.4f}; p-Value - {inside_p_value:.4f}"
                  f"\n\t\t\t*** Outside: Statistic - {outside_statistic:.4f}; p-Value - {outside_p_value:.4f}")

        print(f"\t* Individual Results: ")
        for count_index in range(0, len(set_divisions[sets[set_index]])):
            print(f"\t\t** {set_divisions[sets[set_index]][count_index]}: "
                  f"{set_inside_counts[sets[set_index]][count_index]} Inside Tags, "
                  f"{set_outside_counts[sets[set_index]][count_index]} Outside Tags")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("input_directory", type=str, help=SplitMessage.INPUT_DIRECTORY)
    parser.add_argument("output_directory", type=str, help=SplitMessage.OUTPUT_DIRECTORY)
    parser.add_argument("--match-directory", type=str, default=None, help=SplitMessage.MATCH_DIRECTORY)
    parser.add_argument("--sets", nargs="*", type=str, default=DEFAULT_SPLITS, help=SplitMessage.SETS)
    parser.add_argument("--ratios", nargs="*", type=float, default=[.7, .1, .1, .1], help=SplitMessage.RATIOS)
    parser.add_argument(
        "--loader", type=str, choices=DEFINED_DATASETS, default=DefinedParallelismDataset.ASP,
        help=GenericMessage.LOADER
    )
    parser.add_argument("--stratum-count", type=int, default=2, help=GenericMessage.STRATUM_COUNT)

    args: Namespace = parser.parse_args()

    # We assure the validity of the arguments...
    if not (path.exists(args.input_directory) and path.isdir(args.input_directory)):
        raise ValueError("The input filepath should be a directory containing tagged files representing the data.")
    elif not (path.exists(args.output_directory) and path.isdir(args.output_directory)):
        raise ValueError("The output filepath should be a pre-existing directory.")
    elif args.match_directory is not None and not path.isdir(args.output_directory):
        raise ValueError("The matching directory should be a pre-existing directory.")

    if len(args.sets) < 2 or len(args.ratios) < 2:
        raise ValueError("There must be at least two sets and two ratios into which the data will be split.")
    elif len(args.sets) != len(args.ratios):
        raise ValueError("The number of sets and the number of ratios must be the same.")

    tag_kwargs: dict[str, str] = \
        {"link": TagLink.TOKEN_DISTANCE, "stratum_count": args.stratum_count, "tagset": Tagset.BIO}
    loading_kwargs: dict[str, Any] = {"collection_format": CollectionFormat.DOCUMENT, "tagging_kwargs": tag_kwargs}
    if args.loader == DefinedParallelismDataset.ASP:
        loading_kwargs["tokenizer"] = LatinWordTokenizer()

    input_filenames: list[str] = listdir(args.input_directory)
    _, dataset_loader = get_dataset(args.loader)
    count_triples: list[tuple[int, int, str]] = []
    for filename_index, filename in tqdm(enumerate(input_filenames), desc="Per-File Tag Counting"):
        if filename.endswith("xml"):
            outside_count, inside_count = \
                count_nested_tags(args.input_directory, filename, filename_index, dataset_loader, loading_kwargs)
        else:
            raise ValueError(f"The file type for the file <{filename}> is not supported. Please try again.")
        tag_count_triple: tuple[int, int, str] = (inside_count, outside_count, filename)
        count_triples.append(tag_count_triple)

    # We sort the counts, ordering them from maximum to minimum.
    count_triples.sort(reverse=True)

    if args.match_directory is None:
        file_directory: dict[str, list[str]] = split_by_tag_mse(count_triples, args.sets, args.ratios)
    else:
        file_directory = split_by_matching_directory(count_triples, args.sets, args.ratios, args.match_directory)

    for directory_name in args.sets:
        new_directory: str = f"{args.output_directory}/{directory_name}"
        if not path.exists(new_directory):
            mkdir(new_directory)

        for filename in file_directory[directory_name]:
            copy(f"{args.input_directory}/{filename}", f"{new_directory}/{filename}")
