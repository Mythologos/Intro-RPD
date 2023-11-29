from itertools import chain
from sys import stdout
from typing import Any, Callable, Optional, TextIO, Union

from torch.nn import Module
from tqdm import tqdm

from utils.data.loaders.constants import ParallelismDataset
from utils.stats.loaders import compute_tag_scoring_structure
from utils.stats.structures.match_box import MatchBox

TokenComposer = dict[str, list[tuple[list[str], int]]]
TagComposer = dict[str, list[tuple[list[list[str]], int]]]
TagEvaluationCollection = list[tuple[Union[int, str], list[list[str]], list[list[str]]]]


def evaluate(evaluating_model: Module, evaluating_device: str, dataset: ParallelismDataset,
             evaluation_arguments: dict[str, Any], file_arguments: dict[str, Union[str, TextIO, None]],
             result_file_alias: str, use_tqdm: bool, evaluation_partition: Optional[str] = None) -> dict[str, Any]:
    evaluating_model.eval()

    outputs: dict[str, Any] = {
        "precision": None,
        "recall": None,
        "f1": None,
        "scoring_structure": None
    }

    predictions: TagComposer = {}
    ground_truth_values: TagComposer = {}
    collection_units: TokenComposer = {}
    evaluation_partition: str = evaluation_partition if evaluation_partition is not None \
        else evaluation_arguments["evaluation_partition"]

    displayed_results_count: int = 0
    sample_string: str = ""
    for unit_tokens, gold_tags, identifiers in tqdm(dataset[evaluation_partition], file=stdout, disable=not use_tqdm):
        # We turn the input sentences and tags into tensors.
        unit_tensor, unit_kwargs = evaluating_model.prepare_word_sequence(unit_tokens)
        unit_tensor = unit_tensor.to(evaluating_device)

        _, tag_index_sequence = evaluating_model(unit_tensor, **unit_kwargs)
        predicted_tags: list[list[str]] = evaluating_model.revert_tags(tag_index_sequence, len(gold_tags))

        (collection_id, partition_id) = identifiers
        if predictions.get(collection_id, None) is None:
            predictions[collection_id] = []
            ground_truth_values[collection_id] = []

        for stratum in range(0, len(gold_tags)):
            assert len(predicted_tags[stratum]) == len(gold_tags[stratum])

        predictions[collection_id].append((predicted_tags, partition_id))
        ground_truth_values[collection_id].append((gold_tags, partition_id))

        if file_arguments.get("output_directory", None) is not None:
            if collection_units.get(collection_id, None) is None:
                collection_units[collection_id] = []
            collection_units[collection_id].append((unit_tokens, partition_id))

        # We display one of the values from the input to get a sense of how it's performing.
        # We assure that the values displayed have branch information in them, as they're more "interesting."
        if displayed_results_count < evaluation_arguments["result_display_count"] and \
                (len(set(gold_tags[0])) > 1 or len(set(predicted_tags[0])) > 1):
            current_sample_string: str = "Sample ({0}, {1}):" \
                                         "\n\t* Predictions: {2}" \
                                         "\n\t* Gold: {3}" \
                                         "\n\t* Text: {4}" \
                                         "\n\n".format(*identifiers, predicted_tags, gold_tags, unit_tokens)

            sample_string += current_sample_string

            displayed_results_count += 1

    if sample_string != "":
        if evaluation_arguments["print_style"] == "all":
            print(sample_string, flush=True)

        if file_arguments.get(result_file_alias, None) is not None:
            file_arguments[result_file_alias].write(sample_string)

    collection_results_string: str = ""
    stratum_count: int = evaluation_arguments["tagging_kwargs"]["stratum_count"]
    tag_collection: TagEvaluationCollection = \
        compose_evaluation_collection(predictions, ground_truth_values, stratum_count)
    if file_arguments.get("output_directory", None) is not None:
        organize_nested_collection(collection_units)

    total_scoring_results: MatchBox = MatchBox()
    for (collection_id, collection_predictions, collection_ground_truths) in tag_collection:
        scoring_result: MatchBox = compute_tag_scoring_structure(
            evaluation_arguments["scoring_mode"], prediction_tags=collection_predictions,
            gold_tags=collection_ground_truths, link=evaluation_arguments["tagging_kwargs"]["link"]
        )

        if file_arguments.get(result_file_alias, None) is not None:
            result_statistics: str = scoring_result.get_statistics_display()
            collection_results_string += f"Results for Document {collection_id}:\n" \
                                         f"{result_statistics}\n\n"

        if file_arguments.get("output_directory", None) is not None:
            output_filepath: str = f"{file_arguments['output_directory']}/{collection_id}_results.txt"
            output_file: TextIO = open(output_filepath, encoding="utf-8", mode="w+")
            write_tagged_output_file(output_file, collection_units, predictions, ground_truth_values, collection_id)
            output_file.close()

        total_scoring_results += scoring_result

    if file_arguments.get(result_file_alias, None) is not None:
        summative_results_statistics: str = total_scoring_results.get_statistics_display()
        collection_results_string += f"Results for Full Collection:\n" \
                                     f"{summative_results_statistics}\n\n"
        file_arguments[result_file_alias].write(collection_results_string)

    outputs["precision"] = total_scoring_results.calculate_precision()
    outputs["recall"] = total_scoring_results.calculate_recall()
    outputs["f1"] = total_scoring_results.calculate_f_score()
    outputs["scoring_structure"] = total_scoring_results

    return outputs


def compose_evaluation_collection(predictions: TagComposer, ground_truth_values: TagComposer,
                                  stratum_count: int) -> TagEvaluationCollection:
    tag_collection: TagEvaluationCollection = []

    organize_nested_collection(predictions)
    organize_nested_collection(ground_truth_values)
    for collection_id in predictions.keys():
        document_predictions: list[list[str]] = []
        document_ground_truths: list[list[str]] = []
        collection_item: tuple[Union[int, str], list[list[str]], list[list[str]]] = \
            (collection_id, document_predictions, document_ground_truths)
        for stratum_number in range(0, stratum_count):
            unit_stratum_predictions = predictions[collection_id]
            unit_stratum_ground_truths = ground_truth_values[collection_id]
            stratum_predictions: list[str] = \
                list(chain.from_iterable([unit[stratum_number] for (unit, unit_index) in unit_stratum_predictions]))
            stratum_ground_truths: list[str] = \
                list(chain.from_iterable([unit[stratum_number] for (unit, unit_index) in unit_stratum_ground_truths]))
            document_predictions.append(stratum_predictions)
            document_ground_truths.append(stratum_ground_truths)
        tag_collection.append(collection_item)

    return tag_collection


def organize_nested_collection(collection: Union[TokenComposer, TagComposer]):
    sorting_lambda: Callable = lambda item: item[-1]
    for collection_id in collection.keys():
        collection[collection_id].sort(key=sorting_lambda)


def write_tagged_output_file(output_file: TextIO, units: TokenComposer, predictions: TagComposer,
                             ground_truth_values: TagComposer, collection_id: str):
    collection_units: list[tuple[list[str], int]] = units[collection_id]
    collection_predictions: list[tuple[list[list[str]], int]] = predictions[collection_id]
    collection_ground_truths: list[tuple[list[list[str]], int]] = ground_truth_values[collection_id]
    for unit_index in range(0, len(collection_units)):
        current_units, _ = collection_units[unit_index]
        current_predictions, _ = collection_predictions[unit_index]
        current_ground_truths, _ = collection_ground_truths[unit_index]
        output_file.write(f"Partition {unit_index + 1}:\n")
        current_line: str = ""
        for token_index in range(0, len(current_units)):
            current_line += f"{current_units[token_index]}/"
            inequality_indicator: str = ""
            for stratum_index in range(0, len(current_ground_truths)):
                current_line += f"{current_predictions[stratum_index][token_index]}#" \
                                f"{current_ground_truths[stratum_index][token_index]}"
                if current_predictions[stratum_index][token_index] != current_ground_truths[stratum_index][token_index]:
                    inequality_indicator += "*"

                if stratum_index < len(current_ground_truths) - 1:
                    current_line += ";;"
            else:
                current_line += inequality_indicator
            current_line += " "
        else:
            current_line = current_line.strip()   # We remove the last space in exchange for a newline character.
            current_line += "\n"
        output_file.write(current_line)
