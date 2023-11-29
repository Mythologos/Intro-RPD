from argparse import ArgumentParser, Namespace
from csv import reader
from os import path
from typing import Any, Sequence

from aenum import NamedConstant
from matplotlib import pyplot
from numpy import mean
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.stats import friedmanchisquare, rankdata
from scikit_posthocs import posthoc_nemenyi_friedman

from utils.cli.constants import DEFAULT_SPLITS
from utils.cli.messages import AnalyzerMessage, GenericMessage
from utils.models.constants import EmbeddingType, EncoderType
from utils.data.tags import TagLink
from utils.stats.constants import ScoringMode, SCORING_MODES

ResultSubset = dict[Sequence[str], list[float]]


class SortingCriterion(NamedConstant):
    EMBEDDING: str = "embedding"
    ENCODER: str = "encoder"
    TAGSET: str = "tagset"
    LINK: str = "link"


SORTING_CRITERIA: Sequence[str] = tuple([criterion for criterion in SortingCriterion])   # type: ignore


SORTING_CRITERION_READABLE_NAMES: dict[str, dict[str, str]] = {
    SortingCriterion.EMBEDDING: {
        EmbeddingType.LEARNED: "Learned [Word]",
        EmbeddingType.WORD: "Word",
        EmbeddingType.LATIN_LEARNED_SUBWORD: "Learned [Subword]",
        EmbeddingType.LATIN_BERT: "Latin BERT",
        EmbeddingType.CHINESE_BERT: "Chinese BERT"
    },
    SortingCriterion.ENCODER: {
        EncoderType.IDENTITY: "None",
        EncoderType.LSTM: "BiLSTM",
    },
    SortingCriterion.LINK: {
        TagLink.TOKEN_DISTANCE: "Token Distance",
        TagLink.BRANCH_DISTANCE: "Branch Distance"
    }
}


def get_results_subset(filename: str, criteria: list[str], metric: str, evaluation_set: str) -> ResultSubset:
    results_subset: ResultSubset = {}
    with open(filename, encoding="utf-8", mode="r") as csv_file:
        csv_reader = reader(csv_file)
        for line_index, line in enumerate(csv_reader):
            if line_index == 0:
                continue
            else:
                current_embedding, current_encoder, current_tagset, current_link, current_metric, current_set, \
                    current_precision, current_recall, current_f1 = line
                if metric == current_metric and evaluation_set == current_set:
                    current_criteria: list[str] = []
                    for criterion in criteria:
                        if criterion == SortingCriterion.EMBEDDING:
                            current_criteria.append(current_embedding)
                        elif criterion == SortingCriterion.ENCODER:
                            current_criteria.append(current_encoder)
                        elif criterion == SortingCriterion.TAGSET:
                            current_criteria.append(current_tagset)
                        elif criterion == SortingCriterion.LINK:
                            current_criteria.append(current_link)
                    current_criteria: Sequence[str] = tuple(current_criteria)
                    if current_criteria not in results_subset:
                        results_subset[current_criteria] = []
                    results_subset[current_criteria].append(float(current_f1))

    return results_subset


def build_ranked_arrays(results_subset: ResultSubset) -> tuple[list[Sequence[str]], list[list[float]], NDArray]:
    original_order: list[Sequence[str]] = []
    ranking_arrays: list[list[float]] = [[]]
    for (key, values) in results_subset.items():
        original_order.append(key)
        for value_index, value in enumerate(values):
            if value_index == len(ranking_arrays):
                ranking_arrays.append([])
            ranking_arrays[value_index].append(value)

    flipped_ranking_arrays: list[list[float]] = [[-1 * value for value in array] for array in ranking_arrays]
    ranked_arrays: NDArray = rankdata(flipped_ranking_arrays, axis=1)
    return original_order, ranking_arrays, ranked_arrays


def perform_multiple_hypothesis_test(original_order: list[Sequence[str]], ranking_arrays: list[list[float]],
                                     ranked_arrays: NDArray, alpha: float):
    desired_p_value: float = 1.0 - alpha
    test_statistic, obtained_p_value = friedmanchisquare(*ranking_arrays)
    output_string: str = "The data {verb} the null hypothesis that samples of the same individuals " \
                         "have the same distribution. ({obtained_p} < {desired_p}; Statistic: {statistic})"

    if obtained_p_value < desired_p_value:
        verb: str = "rejects"
    else:
        verb = "accepts"

    output_string_kwargs: dict[str, Any] = {
        "verb": verb,
        "obtained_p": obtained_p_value,
        "desired_p": round(desired_p_value, ndigits=4),
        "statistic": test_statistic
    }
    print(output_string.format(**output_string_kwargs))

    average_rank_output: str = "Average Ranks ({0} Items):".format(len(original_order))
    average_ranks: NDArray = mean(ranked_arrays, axis=0)
    average_rank_dict: dict[str, float] = {}
    for i in range(0, len(original_order)):
        average_rank_output += f"\n\t* {original_order[i]}: {average_ranks[i]:.4f}"
        average_rank_dict["-".join(original_order[i])] = average_ranks[i].item()
    print(average_rank_output)

    if obtained_p_value < desired_p_value:
        p_values: DataFrame = posthoc_nemenyi_friedman(ranking_arrays)
        print(f"P-Value Table:\n{p_values}")
        for i in range(0, len(original_order)):
            for j in range(i, len(original_order)):
                if p_values.at[i, j] < desired_p_value:
                    print(f"There's statistical significance between {original_order[i]} and {original_order[j]} "
                          f"with value {p_values.at[i, j]}.")


def plot_scores_by_group(results_subset: ResultSubset, metric: str, split: str, criteria: list[str],
                         output_filepath: str):
    points: list[list[float]] = [value for value in results_subset.values()]
    labels: list[str] = []

    figure, axis = pyplot.subplots()

    for key in results_subset.keys():
        readable_subkeys: list[str] = []
        for subkey_index, subkey in enumerate(key):
            if criteria[subkey_index] in SORTING_CRITERION_READABLE_NAMES.keys():
                if subkey in SORTING_CRITERION_READABLE_NAMES[criteria[subkey_index]]:
                    readable_subkey = SORTING_CRITERION_READABLE_NAMES[criteria[subkey_index]][subkey]
                else:
                    readable_subkey: str = subkey.replace("-", " ").title()
            else:
                readable_subkey: str = subkey.replace("-", " ").title()
            readable_subkeys.append(readable_subkey)
        labels.append(";\n".join(readable_subkeys))

    axis.boxplot(
        points,
        sym="x",
        patch_artist=True,
        boxprops={"facecolor": "cornflowerblue", "edgecolor": "midnightblue", "linewidth": 1.5},
        capprops={"color": "midnightblue", "linewidth": 1.25},
        flierprops={"color": "midnightblue", "linewidth": 1.5},
        medianprops={"color": "midnightblue", "linewidth": 1.25},
        whiskerprops={"color": "midnightblue", "linewidth": 1.25},
        labels=labels
    )
    pyplot.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    value_range: list[float] = [value for values in results_subset.values() for value in values]
    min_value: float = min(value_range)
    max_value: float = max(value_range)
    current_value: float = 0.0
    tick_range: list[float] = []
    while current_value <= 1.0:
        if (min_value - .05) < current_value < (max_value + .05):
            tick_range.append(current_value)
        current_value += .05

    axis.yaxis.set_ticks(tick_range)
    axis.set_title(f"{metric.upper()} F1 Scores on the {split.title()} Set (Groups: {'; '.join(criteria)})")
    figure.tight_layout()
    pyplot.savefig(f"{output_filepath}.pdf", backend="pgf")


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.95, help=AnalyzerMessage.ALPHA)
    parser.add_argument("--analysis-type", type=str, required=True, choices=["friedman", "box"], help=AnalyzerMessage.ANALYSIS_TYPE)
    parser.add_argument("--criteria", nargs="+", type=str, choices=SORTING_CRITERIA, help=AnalyzerMessage.CRITERIA)
    parser.add_argument("--input-file", type=str, help=AnalyzerMessage.INPUT_FILE)
    parser.add_argument("--output-filepath", type=str, help=AnalyzerMessage.OUTPUT_FILEPATH)
    parser.add_argument(
        "--scoring-mode", type=str, choices=SCORING_MODES, default=ScoringMode.EXACT_PARALLEL_MATCH,
        help=GenericMessage.SCORING_MODE
    )
    parser.add_argument("--split", type=str, choices=DEFAULT_SPLITS, default="test", help=AnalyzerMessage.SPLIT)
    args: Namespace = parser.parse_args()

    if path.isfile(args.input_file) is False:
        raise ValueError(f"The given input filepath, <{args.input_file}>, is not a valid, existing file.")
    else:
        subset: ResultSubset = get_results_subset(args.input_file, args.criteria, args.metric, args.split)

    if args.analysis_type == "friedman":
        order, unranked_values, ranks = build_ranked_arrays(subset)
        perform_multiple_hypothesis_test(order, unranked_values, ranks, args.alpha)
    elif args.analysis_type == "box":
        plot_scores_by_group(subset, args.metric, args.split, args.criteria, args.output_filepath)
