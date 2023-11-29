from itertools import chain
from os import listdir, path
from typing import Any, Optional, Type
from xml.etree import ElementTree as ETree
from xml.etree.ElementTree import Element, ElementTree

from cltk.alphabet.lat import JVReplacer
from cltk.lemmatize import LatinBackoffLemmatizer
from gensim.models import KeyedVectors, Word2Vec
from numpy import zeros
from numpy.typing import NDArray

from utils.data.constants import BranchRepresentation, DefinedParallelismDataset, ParallelismDirectory
from utils.data.loaders.loader import BaseTagLoader
from utils.data.loaders.constants import ParallelismDataset, UnitCollection
from utils.data.loaders.asp import SermonTagLoader
from utils.data.loaders.pse import AugmentedPaibiTagLoader
from utils.data.tags import BIOTag
from utils.data.tokens import UNK_TOKEN


DATASETS: dict[str, tuple[str, Type[BaseTagLoader]]] = {
    DefinedParallelismDataset.ASP: ("data/asp/splits", SermonTagLoader),
    DefinedParallelismDataset.PSE: ("data/pse/splits", AugmentedPaibiTagLoader),
}


def collect_word_vocabulary_data(words: list[str], word2freq: dict[str, int], word2index: dict[str, int],
                                 lemmatizer: Optional[LatinBackoffLemmatizer] = None, **kwargs):
    for word in words:
        if lemmatizer is not None:
            processed_word: str = lemmatizer.lemmatize([word])[0][-1]
        else:
            processed_word: str = word

        if kwargs["embedding_filepath"] is None and word2index.get(processed_word, None) is None:
            word2index[processed_word] = len(word2index)
            word2freq[processed_word] = 1
        elif kwargs["embedding_filepath"] is None and word2index.get(processed_word, None) is not None:
            word2freq[processed_word] += 1


def define_data(dataset_directory: str, dataset_loader: BaseTagLoader,
                data_splits: list[str], required_partitions: list[str],
                loading_kwargs: dict[str, Any]) -> ParallelismDataset:
    tagged_data: ParallelismDataset = {}

    for split in data_splits:
        if split in required_partitions:
            loading_kwargs["split"] = split
            tagged_data[split] = []
            split_filepath: str = f"{dataset_directory}/{split}"

            if path.isdir(split_filepath) is True:
                split_filenames: list[str] = listdir(split_filepath)
                for filename_id, filename in enumerate(split_filenames, start=1):
                    units: UnitCollection = dataset_loader(split_filepath, filename, filename_id, loading_kwargs)
                    tagged_data[split].extend(units)
            else:
                raise ValueError(f"The path <{split_filepath}> is not valid.")

    return tagged_data


def compose_class_reference_parallelism_directory(reference_filepath: str, representation: str) -> ParallelismDirectory:
    directory: ParallelismDirectory = {}
    tree: ElementTree = ETree.parse(reference_filepath)
    root: Element = tree.getroot()

    current_word_index: int = 0
    for section in root:
        current_word_index = process_section(section, directory, representation, current_word_index)

    return directory


def process_section(element: Element, parallelism_directory: dict, branch_representation: str, word_index: int) -> int:
    element_tag: str = element.tag
    element_text: str = element.text
    tokenized_text: list[str] = get_word_tokens(element_text)

    if element_tag == "parallelism":
        parallelism_id: int = int(element.attrib["id"])
        element_full_text: str = "".join(element.itertext())
        tokenized_full_text: list[str] = get_word_tokens(element_full_text)

        if parallelism_directory.get(parallelism_id, None) is None:
            if branch_representation == BranchRepresentation.TUPLE:
                parallelism_directory[parallelism_id] = set()
            elif branch_representation == BranchRepresentation.SET:
                parallelism_directory[parallelism_id] = []
            else:
                raise ValueError(f"The branch representation <{branch_representation}> is not supported.")

        branch_end_index: int = word_index + len(tokenized_full_text)
        if branch_representation == BranchRepresentation.TUPLE:
            branch_indices: tuple[int, int] = (word_index, branch_end_index)
            parallelism_directory[parallelism_id].add(branch_indices)
        elif branch_representation == BranchRepresentation.SET:
            branch_token_indices: set[int] = set(range(word_index, branch_end_index))
            parallelism_directory[parallelism_id].append(branch_token_indices)   # type: ignore
        else:
            raise ValueError(f"The branch representation <{branch_representation}> is not supported.")
    elif element_tag != "section":
        raise ValueError(f"Unrecognized tag: {element_tag}. Please try again.")

    # We increment by the text of the element, since there may be internal elements we must yet pass over.
    word_index += len(tokenized_text)

    for child in element:
        word_index = process_section(child, parallelism_directory, branch_representation, word_index)

    element_tail: str = element.tail
    tokenized_tail: list[str] = get_word_tokens(element_tail)
    word_index += len(tokenized_tail)

    return word_index


def get_word_tokens(raw_xml_text: str) -> list[str]:
    tokenized_lines: list[list[str]] = [
        line.strip().split() for line in raw_xml_text.split() if len(line.strip().split()) > 0
    ]
    tokens: list[str] = [token for token in chain.from_iterable(tokenized_lines)]
    return tokens


def define_vocabulary_structures(dataset_partitions: ParallelismDataset, lemmatizer: Optional[LatinBackoffLemmatizer],
                                 **kwargs) -> dict[str, dict]:
    words_to_frequency: dict[str, int] = {}
    words_to_indices: dict[str, int] = {UNK_TOKEN: 0}
    tags_to_indices: dict[str, int] = {BIOTag.START.value: 0, BIOTag.STOP.value: 1}
    vocabulary_structures: dict[str, dict] = {"words_to_indices": words_to_indices, "tags_to_indices": tags_to_indices}

    if kwargs["embedding_filepath"] is None:
        vocabulary_structures["words_to_frequency"] = words_to_frequency

    for (words, tags, identifiers) in dataset_partitions["training"]:
        collect_word_vocabulary_data(words, words_to_frequency, words_to_indices, lemmatizer, **kwargs)

        for stratum in tags:
            for tag in stratum:
                if tags_to_indices.get(tag, None) is None:
                    tags_to_indices[tag] = len(tags_to_indices)

    indices_to_tags: dict[int, str] = {index: tag for tag, index in tags_to_indices.items()}
    vocabulary_structures["indices_to_tags"] = indices_to_tags

    if kwargs.get("embedding_filepath", None) is not None:
        load_embeddings(kwargs["embedding_filepath"], vocabulary_structures)

    return vocabulary_structures


def get_dataset(dataset_name: str) -> tuple[str, BaseTagLoader]:
    try:
        dataset_directory, dataset_loader_class = DATASETS[dataset_name]
        dataset_loader: BaseTagLoader = dataset_loader_class()
    except KeyError:
        raise ValueError(f"The dataset name <{dataset_name}> is not valid. Please try again.")
    return dataset_directory, dataset_loader


def get_lemmatizer(should_lemmatize: bool) -> Optional[LatinBackoffLemmatizer]:
    chosen_lemmatizer: Optional[LatinBackoffLemmatizer] = None
    if should_lemmatize is True:
        chosen_lemmatizer = LatinBackoffLemmatizer()
    return chosen_lemmatizer


def load_embeddings(embedding_filepath: str, vocabulary_structures: dict[str, dict]):
    replacer: JVReplacer = JVReplacer()
    latin_word2vec: KeyedVectors = Word2Vec.load(embedding_filepath).wv
    word_embeddings: dict[str, NDArray] = {}

    # As in Burns et al. 2021, we preprocess the word2vec keys to reduce ambiguity between u/v and i/j.
    #   In this way, more lemmas can be captured, as we reduce variation.
    for (key, index) in latin_word2vec.key_to_index.items():
        if key != UNK_TOKEN:
            processed_key: str = replacer.replace(key)
        else:
            processed_key = key

        # If there was an embedding for the un-replaced version, then we keep that.
        # Otherwise, we add the replaced version.
        if processed_key not in word_embeddings:
            word_embeddings[processed_key] = latin_word2vec[index]
    else:
        if UNK_TOKEN not in word_embeddings:
            word_embeddings[UNK_TOKEN] = zeros(latin_word2vec[0].shape, dtype="float32")   # type: ignore

    vocabulary_structures["word_embeddings"] = word_embeddings


def compute_kneser_ney_estimation(vocabulary_structures: dict[str, dict]) -> float:
    if vocabulary_structures.get("words_to_frequency") is not None:
        singleton_count: int = \
            len([word for word, frequency in vocabulary_structures["words_to_frequency"].items() if frequency == 1])
        doubleton_count: int = \
            len([word for word, frequency in vocabulary_structures["words_to_frequency"].items() if frequency == 2])
        kneser_ney_estimate: float = singleton_count / (singleton_count + 2 * doubleton_count)
    else:
        raise ValueError("The word-to-frequency dictionary was not found. Please check the selected CLI arguments.")

    return kneser_ney_estimate
