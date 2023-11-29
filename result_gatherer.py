from argparse import ArgumentParser, Namespace
from collections import deque
from os import path, walk
from re import search
from typing import Deque, Match, Union

from utils.cli.messages import GatherMessage
from utils.models.constants import EmbeddingType, EncoderType

MODEL_RESULT_LINE_REGEX: str = r"(?P<statistic>\t*[\s]+(?P<name>[\w]+):[\s]+(?P<quantity>[\d].[\d]+))"


def collect_files(starting_directories: list[str], file_type: str, file_regex: str, subdirectory_regex: str) -> \
        list[list[str]]:
    filepaths: list[list[str]] = []
    directory_queue: Deque[str] = deque(starting_directories)
    while len(directory_queue) > 0:
        filepaths.append([])

        current_directory: str = directory_queue.popleft()
        for (entry_path, entry_name, entry_files) in walk(current_directory):
            if subdirectory_regex is not None and search(subdirectory_regex, entry_path) is not None:
                continue
            else:
                for entry_file in entry_files:
                    if entry_file.endswith(file_type) is True and \
                            (file_regex is None or search(file_regex, entry_file)):
                        filepaths[-1].append(f"{entry_path}/{entry_file}")
                    else:
                        continue

    return filepaths


def aggregate_text_results(starting_directories: list[str], text_filepaths: list[list[str]], output_filepath: str):
    with open(output_filepath, encoding="utf-8", mode="w+") as output_file:
        output_file.write("embedding,encoder,tagset,link,metric,set,precision,recall,f1\n")
        for index in range(0, len(starting_directories)):
            current_directory: str = starting_directories[index]
            if "/" in current_directory:
                *_, current_directory = current_directory.rsplit("/", maxsplit=1)

            current_filepaths: list[str] = text_filepaths[index]
            current_embedding, current_encoder = get_model_information(current_directory)

            for current_filepath in current_filepaths:
                filepath, extension = current_filepath.rsplit(".", maxsplit=1)
                if "/" in filepath:
                    _, filename = filepath.rsplit("/", maxsplit=1)
                _, tagset, link, evaluation_set, metric = filename.split("-")

                file_data: list[str] = []
                with open(current_filepath, encoding="utf-8", mode="r") as current_text_file:
                    full_collection_bool: bool = False
                    for line in current_text_file:
                        if full_collection_bool is False and "Results for Full Collection:" in line:
                            full_collection_bool = True
                        elif full_collection_bool is True:
                            line_match: Match = search(MODEL_RESULT_LINE_REGEX, line)
                            if line_match is not None:
                                file_data.append(line_match.group("quantity"))
                        else:
                            continue
                assert len(file_data) == 3
                file_results: str = ",".join(file_data)
                output_file.write(
                    f"{current_embedding},{current_encoder},{tagset},{link},{metric},{evaluation_set},{file_results}\n"
                )


def get_model_information(directory_name: str) -> tuple[str, str]:
    if directory_name == "latin-bert":
        embedding_name: str = EmbeddingType.LATIN_BERT
        encoding_name: str = EncoderType.IDENTITY
    elif directory_name == "latin-bert-bilstm":
        embedding_name = EmbeddingType.LATIN_BERT
        encoding_name = EncoderType.LSTM
    elif directory_name == "latin-bert-transformer":
        embedding_name = EmbeddingType.LATIN_BERT
        encoding_name = EncoderType.TRANSFORMER
    elif directory_name == "latin-le-bilstm":
        embedding_name = EmbeddingType.LEARNED
        encoding_name = EncoderType.LSTM
    elif directory_name == "latin-we-bilstm":
        embedding_name = EmbeddingType.WORD
        encoding_name = EncoderType.LSTM
    elif directory_name == "latin-le-transformer":
        embedding_name = EmbeddingType.LATIN_LEARNED_SUBWORD
        encoding_name = EncoderType.TRANSFORMER
    elif directory_name == "chinese-bert":
        embedding_name = EmbeddingType.CHINESE_BERT
        encoding_name = EncoderType.IDENTITY
    elif directory_name == "chinese-bert-bilstm":
        embedding_name = EmbeddingType.CHINESE_BERT
        encoding_name = EncoderType.LSTM
    elif directory_name == "chinese-bert-transformer":
        embedding_name = EmbeddingType.CHINESE_BERT
        encoding_name = EncoderType.TRANSFORMER
    else:
        raise ValueError(f"The directory of name <{directory_name}> is not currently supported.")
    return embedding_name, encoding_name


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("directories", type=str, nargs="+", help=GatherMessage.DIRECTORIES)
    parser.add_argument("--file-regex", type=str, default=None, help=GatherMessage.FILE_REGEX)
    parser.add_argument("--file-type", type=str, choices=["txt"], default="txt", help=GatherMessage.FILE_TYPE)
    parser.add_argument("--output-file", type=str, default="output.txt", help=GatherMessage.OUTPUT_FILE)
    parser.add_argument("--subdirectory-regex", type=str, default=None, help=GatherMessage.SUBDIRECTORY_REGEX)
    args: Namespace = parser.parse_args()

    for directory in args.directories:
        if path.isdir(directory) is False:
            raise ValueError(f"The path <{directory}> not a valid directory.")

    prefixed_file_type: str = f".{args.file_type}"
    collected_filepaths: Union[list[str], list[list[str]]] = \
        collect_files(args.directories, prefixed_file_type, args.file_regex, args.subdirectory_regex)

    if args.file_type == "txt":
        aggregate_text_results(args.directories, collected_filepaths, args.output_file)
    else:
        raise ValueError(f"The file type <{args.file_type}> currently does not have an analyzer.")
