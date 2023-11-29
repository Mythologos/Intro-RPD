from __future__ import annotations

from argparse import ArgumentParser, Namespace
from os import listdir, path
from sys import stdout
from typing import Any, Sequence, TextIO, Union

from aenum import NamedConstant
from cltk.tokenizers.lat.lat import LatinWordTokenizer
from matplotlib import colormaps, pyplot
from matplotlib.colors import Colormap, LogNorm
from natsort import natsorted
from networkx import DiGraph, write_graphml
from numpy import zeros
from numpy.typing import NDArray
from tqdm import tqdm

from utils.cli.messages import BIOMessage, GenericMessage
from utils.data.loaders.constants import CollectionFormat, COLLECTIONS
from utils.data.loaders.loader import BaseTagLoader
from utils.data.interface import get_dataset, DefinedParallelismDataset, UnitCollection
from utils.data.tags import BIOTag, TagLink, Tagset, LINKS, TAGS, TAGSETS

ADJACENCY_LIST_FORMAT: str = "AdjacencyList ({0} Node Types, {1} Edge Types):" \
                             "\n\t* List:\n{2}" \
                             "\n\t* Node Types: {3}" \
                             "\n\t* Edge Frequencies:\n{4}"


class TagMapping(NamedConstant):
    LINKED: str = "linked"


TAG_MAPPINGS: Sequence[str] = (TagMapping.LINKED,)


class AdjacencyList:
    def __init__(self):
        self.adjacency_list: dict[str, set[str]] = {}
        self.nodes: set[str] = set()
        self.edge_frequencies: dict[tuple[str, str], int] = {}

    def __add__(self, other):
        new_adjacency_list: AdjacencyList = AdjacencyList()
        new_adjacency_list._sum_adjacency_lists(self)
        new_adjacency_list._sum_adjacency_lists(other)
        return new_adjacency_list

    def __iadd__(self, other: AdjacencyList):
        self._sum_adjacency_lists(other)
        return self

    def __str__(self):
        node_count: int = self.get_node_type_count()
        edge_count: int = self.get_edge_type_count()

        list_representation: str = ""
        internal_list: list[tuple[str, set[str]]] = [(key, value) for (key, value) in self.adjacency_list.items()]
        internal_list = natsorted(internal_list)
        for (key, values) in internal_list:
            if len(values) > 0:
                sorted_values = natsorted(values)
                list_representation += f"\t\t- {key}: " + ", ".join(sorted_values) + "\n"
            else:
                list_representation += f"\t\t- {key}: --\n"
        else:
            list_representation = list_representation.rstrip()

        node_list: list[str] = list(self.nodes)
        node_list = natsorted(node_list)
        node_representation: str = ", ".join(node_list)

        edge_frequency_representation: str = ""
        edge_frequency_list: list[tuple[tuple[str, str], int]] = \
            [(key, value) for (key, value) in self.edge_frequencies.items()]
        edge_frequency_list = natsorted(edge_frequency_list)
        for (edge, frequency) in edge_frequency_list:
            edge_frequency_representation += f"\t\t- {edge}: {frequency}\n"
        else:
            edge_frequency_representation = edge_frequency_representation.rstrip()

        full_adjacency_list_string: str = ADJACENCY_LIST_FORMAT.format(
            node_count, edge_count, list_representation, node_representation, edge_frequency_representation
        )
        return full_adjacency_list_string

    def _sum_adjacency_lists(self, other: AdjacencyList):
        for (source, destinations) in other.adjacency_list.items():
            if self.adjacency_list.get(source, None) is not None:
                self.adjacency_list[source].update(destinations)
            else:
                self.adjacency_list[source] = destinations

        self.nodes.update(other.nodes)

        for (edge, frequency) in other.edge_frequencies.items():
            if self.edge_frequencies.get(edge, None) is not None:
                self.edge_frequencies[edge] += frequency
            else:
                self.edge_frequencies[edge] = frequency

    def add_node(self, node: str):
        if self.adjacency_list.get(node, None) is None:
            self.adjacency_list[node] = set()
        self.nodes.add(node)

    def add_edge(self, source: str, destination: str):
        if source not in self.nodes or source not in self.adjacency_list:
            raise ValueError(f"The node <{source}> is not in the list's nodes or adjacency list.")
        elif destination not in self.nodes or destination not in self.adjacency_list:
            raise ValueError(f"The node <{source}> is not in the list's nodes or adjacency list.")

        self.adjacency_list[source].add(destination)

        edge: tuple[str, str] = (source, destination)
        if self.edge_frequencies.get(edge, None) is None:
            self.edge_frequencies[edge] = 0
        self.edge_frequencies[edge] += 1

    def map_nodes(self, sources: list[str], target: str):
        for source in sources:
            self.nodes.remove(source)
            self._remove_self_loops(source, target)
            destinations = self.adjacency_list[source]
            for destination in destinations:
                self._remove_out_edges(source, destination, target)
            del self.adjacency_list[source]

            self._remove_in_edges(source, target)

        self.nodes.add(target)

    def _remove_out_edges(self, source: str, destination: str, target: str):
        if target not in self.adjacency_list:
            self.adjacency_list[target] = set()

        self.adjacency_list[target].add(destination)
        old_edge_frequency: int = self._delete_edge(source, destination)
        new_edge: tuple[str, str] = (target, destination)
        self._add_new_frequency(new_edge, old_edge_frequency)

    def _remove_in_edges(self, destination: str, target: str):
        for source in self.adjacency_list:
            if destination in self.adjacency_list[source]:
                self.adjacency_list[source].remove(destination)
                self.adjacency_list[source].add(target)
                old_edge_frequency: int = self._delete_edge(source, destination)
                new_edge: tuple[str, str] = (source, target)
                self._add_new_frequency(new_edge, old_edge_frequency)

    def _remove_self_loops(self, node: str, target: str):
        if node in self.adjacency_list[node]:
            self.adjacency_list[node].remove(node)

            if target not in self.adjacency_list.keys():
                self.adjacency_list[target] = set()
            self.adjacency_list[target].add(target)

            old_edge_frequency: int = self._delete_edge(node, node)
            new_edge: tuple[str, str] = (target, target)
            self._add_new_frequency(new_edge, old_edge_frequency)

    def _delete_edge(self, first: str, second: str) -> int:
        old_edge: tuple[str, str] = (first, second)
        old_edge_frequency: int = self.edge_frequencies[old_edge]
        del self.edge_frequencies[old_edge]
        return old_edge_frequency

    def _add_new_frequency(self, new_edge: tuple[str, str], old_edge_frequency: int):
        if self.edge_frequencies.get(new_edge, None) is None:
            self.edge_frequencies[new_edge] = old_edge_frequency
        else:
            self.edge_frequencies[new_edge] += old_edge_frequency

    def get_node_type_count(self) -> int:
        return len(self.nodes)

    def get_edge_type_count(self) -> int:
        return sum([len(value) for value in self.adjacency_list.values()])

    def get_edge_count(self) -> int:
        return sum([frequency for frequency in self.edge_frequencies.values()])

    def get_styled_edge_frequencies(self, save_style: str) -> tuple[dict[tuple[str, str], float], type]:
        if save_style == "counts":
            edge_frequencies: dict[tuple[str, str], int] = self.edge_frequencies
            frequency_type: type = int
        elif save_style == "ratios":
            total_edges: int = sum(self.edge_frequencies.values())
            edge_frequencies: dict[tuple[str, str], float] = {}
            for (edge, frequency) in self.edge_frequencies.items():
                edge_frequencies[edge] = (frequency / total_edges)
            frequency_type: type = float
        else:
            raise ValueError(f"The save style <{save_style}> is not supported.")
        return edge_frequencies, frequency_type

    def save_as_graphml(self, save_filepath: str, save_style: str):
        directed_graph: DiGraph = DiGraph()

        edges_with_labels: list[tuple[str, str]] = list(self.edge_frequencies.keys())
        node_mapping: dict[str, int] = {}

        node_list: list[str] = list(self.nodes)
        for node_index, node in enumerate(node_list, start=0):
            node_mapping[node] = node_index
            directed_graph.add_node(node_index, label=node)

        edge_frequencies, edge_type = self.get_styled_edge_frequencies(save_style)

        for edge in edges_with_labels:
            first, second = edge
            directed_graph.add_edge(node_mapping[first], node_mapping[second], frequency=edge_frequencies[edge])

        write_graphml(directed_graph, f"{save_filepath}.graphml")

    # Code below used the matplotlib documentation as a reference.
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    def save_as_heatmap(self, save_filepath: str, save_style: str, save_order: list[str]):
        if len(save_order) == len(list(self.nodes)) == len(set(save_order).intersection(self.nodes)):
            sorted_nodes: list[str] = save_order
        elif len(save_order) == 0:
            sorted_nodes: list[str] = natsorted(list(self.nodes)) if not save_order else save_order
        else:
            raise ValueError(f"An invalid save order was given. Save Order: {save_order}; Nodes: {list(self.nodes)}")

        edge_frequencies, edge_type = self.get_styled_edge_frequencies(save_style)
        heatmap: NDArray = zeros((len(sorted_nodes), len(sorted_nodes)), dtype=edge_type)

        for source_index, source in enumerate(sorted_nodes, 0):
            for destination_index, destination in enumerate(sorted_nodes, 0):
                current_edge: tuple[str, str] = (source, destination)
                current_frequency: float = edge_frequencies.get(current_edge, edge_type(0))
                heatmap[source_index][destination_index] = round(current_frequency, 4)

        # This starts to work, but I'd like more custom scaling.
        # We want "none" and "more than one" to be differentiated,
        # and we want to handle the outlier on the other end--the transitions from O to O.

        node_range = list(range(0, len(sorted_nodes)))
        figure, axis = pyplot.subplots()

        colormap: Colormap = colormaps['Blues']
        colormap.set_under(color='#f7fbff')
        mean_frequency: float = heatmap.mean().item()
        color_norm: LogNorm = LogNorm()

        tag_labels: list[str] = [node if node != BIOTag.LINKED_BEGINNING.value else "B\u03bb" for node in sorted_nodes]
        axis.imshow(heatmap, cmap=colormap, norm=color_norm)
        axis.set_xticks(node_range, rotation=45, ha="right", rotation_mode="anchor", labels=tag_labels, fontsize=12)
        axis.set_yticks(node_range, labels=tag_labels, fontsize=12)
        for source_index in node_range:
            for destination_index in node_range:
                color: str = "k" if heatmap[destination_index, source_index] < mean_frequency else "w"
                heatmap_value: int = heatmap[destination_index, source_index].item()
                if heatmap_value >= 1000:
                    if heatmap_value < 10000:
                        converted_heatmap_value: str = f"{str(heatmap_value)[0:1]}.{str(heatmap_value)[1:2]}"
                    elif heatmap_value < 100000:
                        converted_heatmap_value = f"{str(heatmap_value)[0:2]}.{str(heatmap_value)[2:3]}"
                    else:
                        converted_heatmap_value = str(heatmap_value)[:len(str(heatmap_value)) - 3]
                    converted_heatmap_value += "K"
                else:
                    converted_heatmap_value = str(heatmap_value)

                axis.text(
                    source_index, destination_index, converted_heatmap_value,
                    ha="center", va="center", color=color, fontsize=10
                )

        axis.set_title("Tag Heatmap", fontsize=16)
        pyplot.ylabel("Preceding Word", fontsize=14)
        pyplot.xlabel("Following Word", fontsize=14)
        pyplot.tight_layout()
        pyplot.savefig(f"{save_filepath}.pdf", bbox_inches="tight")

    def save_as_latex_table(self, save_filepath: str, save_style: str):
        sorted_nodes: list[str] = natsorted(list(self.nodes))
        column_line: str = self._get_latex_column_line(sorted_nodes)
        latex_table: str = r"\begin{tabular}" + column_line + "\n"

        divider_line: str = "\t" + r"\hline" + "\n"
        latex_table += divider_line

        first_row_contents: list[str] = ["PREC / FOLL", *sorted_nodes]
        latex_table += "\t" + " & ".join(first_row_contents) + r" \\" + "\n"

        edge_frequencies, edge_type = self.get_styled_edge_frequencies(save_style)

        for source in sorted_nodes:
            latex_table += divider_line
            edge_frequency_row: list[Union[str, int]] = [source]
            for destination in sorted_nodes:
                edge_frequency: float = edge_frequencies.get((source, destination), edge_type(0))
                edge_frequency_row.append(str(edge_frequency))
            new_row: str = "\t" + " & ".join(edge_frequency_row) + r" \\" + "\n"
            latex_table += new_row

        latex_table += divider_line
        latex_table += r"\end{tabular}"

        with open(f"{save_filepath}_latex.txt", encoding="utf-8", mode="w+") as save_file:
            save_file.write(latex_table)

    @staticmethod
    def _get_latex_column_line(sorted_nodes: list[str]) -> str:
        columns: list[str] = []
        for i in range(0, len(sorted_nodes) + 1):
            columns.append("c")

        column_line: str = "{||" + "|".join(columns) + "||}"
        return column_line


def compute_file_adjacency_list(input_path: str, input_filename: str, backup_id: int, loader: BaseTagLoader,
                                formatting_kwargs: dict[str, Any]) -> AdjacencyList:
    file_adjacency_list: AdjacencyList = AdjacencyList()
    # We load by chunks, although we can throw out much of the information, since we only need the tags.
    units: UnitCollection = loader(input_path, input_filename, backup_id, formatting_kwargs)
    tags_by_unit = [tags for (words, tags, identifiers) in units]
    for stratum in range(0, formatting_kwargs["tagging_kwargs"]["stratum_count"]):
        for tags in tags_by_unit:
            add_unit_adjacencies(tags[stratum], file_adjacency_list)

    return file_adjacency_list


def add_unit_adjacencies(tags: list[str], file_adjacency_list: AdjacencyList):
    tag_index: int = 0
    while tag_index <= len(tags):
        source: str = tags[tag_index - 1] if tag_index > 0 else BIOTag.START.value
        destination: str = tags[tag_index] if tag_index != len(tags) else BIOTag.STOP.value
        file_adjacency_list.add_node(source)
        file_adjacency_list.add_node(destination)
        file_adjacency_list.add_edge(source, destination)
        tag_index += 1


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("input_filepath", type=str, help=BIOMessage.INPUT_FILEPATH)
    parser.add_argument(
        "--collection-format", type=str, choices=COLLECTIONS,
        default=CollectionFormat.SECTION, help=GenericMessage.COLLECTION_FORMAT
    )
    parser.add_argument("--link", choices=LINKS, default=TagLink.TOKEN_DISTANCE, help=GenericMessage.LINK)
    parser.add_argument("--loader", type=get_dataset, default=DefinedParallelismDataset.ASP, help=GenericMessage.LOADER)
    parser.add_argument("--mappings", type=str, nargs="*", choices=TAG_MAPPINGS, default=[], help=BIOMessage.MAPPINGS)
    parser.add_argument("--output-filepath", type=str, help=BIOMessage.OUTPUT_FILEPATH)
    parser.add_argument(
        "--save-formats", type=str, nargs="*",
        choices=["graphml", "heatmap", "latex"], help=BIOMessage.SAVE_FORMATS
    )
    parser.add_argument(
        "--save-frequency-style", type=str, choices=["counts", "ratios"],
        default="counts", help=BIOMessage.SAVE_FREQUENCY
    )
    parser.add_argument("--save-order", type=str, nargs="*", choices=TAGS, default=[], help=BIOMessage.SAVE_ORDER)
    parser.add_argument("--save-path", type=str, default=None, help=BIOMessage.SAVE_PATH)
    parser.add_argument("--stratum-count", type=int, default=1, help=GenericMessage.STRATUM_COUNT)
    parser.add_argument("--tagset", choices=TAGSETS, default=Tagset.BIO, help=GenericMessage.TAGSET)
    args: Namespace = parser.parse_args()

    if not path.exists(args.input_filepath) or not path.isdir(args.input_filepath):
        raise ValueError("Invalid filepath given for input directory. Please try again.")

    tagging_kwargs: dict[str, str] = {"link": args.link, "stratum_count": args.stratum_count, "tagset": args.tagset}
    _, dataset_loader = args.loader
    loading_kwargs: dict[str, Any] = {
        "collection_format": args.collection_format,
        "tagging_kwargs": tagging_kwargs,
        "tokenizer": LatinWordTokenizer()
    }

    filenames: list[str] = listdir(args.input_filepath)
    adjacency_lists: list[AdjacencyList] = []
    for filename_id, filename in tqdm(enumerate(filenames), file=stdout):
        current_adjacency_list: AdjacencyList = \
            compute_file_adjacency_list(args.input_filepath, filename, filename_id, dataset_loader, loading_kwargs)
        adjacency_lists.append(current_adjacency_list)
    else:
        total_adjacency_list: AdjacencyList = AdjacencyList()
        for adjacency_list in adjacency_lists:
            total_adjacency_list += adjacency_list

    for mapping in args.mappings:
        if mapping == TagMapping.LINKED:
            mapping_sources = [tag for tag in total_adjacency_list.nodes if tag.startswith("B-")]
            mapping_destination = BIOTag.LINKED_BEGINNING.value
            total_edges_prior: int = total_adjacency_list.get_edge_count()
            total_adjacency_list.map_nodes(mapping_sources, mapping_destination)
            total_edges_posterior: int = total_adjacency_list.get_edge_count()
            assert total_edges_prior == total_edges_posterior

    if args.output_filepath is not None and path.isdir(args.output_filepath):
        raise ValueError(f"The output filepath <{args.output_filepath}> is a directory and cannot be used.")
    elif args.output_filepath is not None:
        output_file: TextIO = open(f"{args.output_filepath}.txt", encoding="utf-8", mode="w+")
        output_file.write(str(total_adjacency_list))
        output_file.close()
    else:
        print(total_adjacency_list)
        print("\n")

    if args.save_formats is None and args.save_path is None:
        pass
    elif args.save_formats is not None and args.save_path is None:
        raise ValueError("The save filepath is not defined, although options are defined.")
    elif args.save_formats is None and args.save_path is not None:
        raise ValueError("The save filepath is defined, although no options are defined.")
    elif path.isdir(args.save_path) is True:
        raise ValueError(f"The path <{args.save_path}> is a directory. "
                         f"Please provide a path to a file (minus an extension).")
    else:
        save_formats: set[str] = set(args.save_formats)
        for save_format in save_formats:
            if save_format == "graphml":
                total_adjacency_list.save_as_graphml(args.save_path, args.save_frequency_style)
            elif save_format == "heatmap":
                total_adjacency_list.save_as_heatmap(args.save_path, args.save_frequency_style, args.save_order)
            elif save_format == "latex":
                total_adjacency_list.save_as_latex_table(args.save_path, args.save_frequency_style)
