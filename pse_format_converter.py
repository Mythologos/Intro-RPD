from argparse import ArgumentParser, Namespace
from collections import deque
from os import listdir, path
from typing import Any, Optional, Union
from xml.etree import ElementTree as ETree
from xml.etree.ElementTree import Element, ElementTree

from utils.cli.messages import PSEMessage
from utils.data.converters.annotator import process_annotation_file
from utils.data.converters.constants import BratAnnotation
from utils.data.loaders.constants import DocumentState, PSEFileGroup, PSEWord, PSE_ENDING_PUNCTUATION, \
    PSE_PARAGRAPH_TEMPLATE, PSE_DESIGNATED_LINE_BEGINNING
from utils.data.loaders.pse import PaibiTag, PAIBI_INNER_TAGS, PAIBI_OUTER_TAGS, PAIBI_BEGINNING_TAGS, PAIBI_MIDDLE_TAGS


def coordinate_files(paibi_directory: str, annotation_directory: str, output_directory: str) -> list[PSEFileGroup]:
    coordinated_files: list[PSEFileGroup] = []
    paibi_filenames: list[str] = listdir(paibi_directory)
    annotation_filenames: list[str] = listdir(annotation_directory)
    for paibi_filename in paibi_filenames:
        paibi_filepath: str = f"{paibi_directory}/{paibi_filename}"
        output_filepath: str = f"{output_directory}/{paibi_filename}"
        paibi_name, paibi_extension = paibi_filename.rsplit(".", maxsplit=1)
        for annotation_filename in annotation_filenames:
            if annotation_filename.endswith(".ann"):
                annotation_name, annotation_extension = annotation_filename.rsplit(".", maxsplit=1)
                if paibi_name == annotation_name:
                    text_filepath: str = f"{annotation_directory}/{annotation_name}.txt"
                    annotation_filepath: str = f"{annotation_directory}/{annotation_name}.ann"

                    coordinated_file_group: PSEFileGroup = \
                        (paibi_filepath, (text_filepath, annotation_filepath), output_filepath)
                    coordinated_files.append(coordinated_file_group)
                    break
        else:
            coordinated_file_group: PSEFileGroup = (paibi_filepath, None, output_filepath)
            coordinated_files.append(coordinated_file_group)

    return coordinated_files


def read_paibi_file(paibi_filepath: str) -> \
        tuple[dict[str, Union[int, str]], Optional[Element], list[PSEWord], list[tuple[int, int]]]:
    paibi_words: list[PSEWord] = []

    tree: ElementTree = ETree.parse(paibi_filepath)
    root: Element = tree.getroot()
    document: Element = root.find("doc")
    document_attributes: dict[str, Union[int, str]] = document.attrib

    document_state: DocumentState = {"branch_tracker": {}, "inner_queue": deque([]), "paibi_sentences": []}
    scrawl: Optional[Element] = None
    for element in document:
        if element.tag == "para":
            read_paibi_paragraph(element, paibi_words, document_state)
        elif element.tag == "scrawl":
            if scrawl is None:
                scrawl = element
            else:
                raise NotImplementedError("There are multiple scrawls in this file, which is not supported.")
        else:
            raise NotImplementedError(f"Currently, this code does not handle the tag <{element.tag}>")

    document_attributes["parallelisms"] = len(document_state["branch_tracker"])
    return document_attributes, scrawl, paibi_words, document_state["paibi_sentences"]


def read_paibi_paragraph(paragraph: Element, paibi_words: list[PSEWord], state: DocumentState):
    branch_tracker: dict[int, int] = state["branch_tracker"]
    for sentence in paragraph:
        sentence_type: Optional[str] = sentence.attrib.get("type", None)
        if sentence.tag == "sent":
            if sentence_type in PAIBI_INNER_TAGS:
                relevant_stack = state["inner_queue"]

                if sentence_type in PAIBI_BEGINNING_TAGS:
                    current_parallelism_id: Optional[int] = len(branch_tracker)
                    relevant_stack.appendleft(current_parallelism_id)
                    branch_tracker[relevant_stack[0]] = 1
                elif sentence_type in PAIBI_MIDDLE_TAGS:
                    current_parallelism_id = relevant_stack[0]
                    branch_tracker[current_parallelism_id] += 1
                else:  # element_type in PAIBI_ENDING_TAGS:
                    current_parallelism_id = relevant_stack[0]
                    branch_tracker[current_parallelism_id] += 1
                    relevant_stack.popleft()
            elif sentence_type == PaibiTag.SINGLE_SENTENCE:
                state["paibi_sentences"].append((int(paragraph.attrib["id"]), int(sentence.attrib["id"])))
                current_parallelism_id = None
            elif sentence_type not in (" ", None, *PAIBI_OUTER_TAGS):
                raise ValueError(f"The tag type <{sentence_type}> is not recognized.")
            else:
                current_parallelism_id = None

            sentence_content: str = sentence.attrib["cont"]

            for word_index, word in enumerate(sentence):
                current_word: str = word.attrib["cont"]
                new_word_attributes: dict[str, Any] = {
                    "word": current_word,
                    "paragraph_index": int(paragraph.attrib["id"]),
                    "sentence_index": int(sentence.attrib["id"]),
                    "word_index": int(word.attrib["id"]),
                    "part_of_speech": word.attrib["pos"],
                    "named_entity": word.attrib["ne"],
                    "dependency_parent": word.attrib["parent"],
                    "dependency_tag": word.attrib["relate"],
                }

                if word_index == 0:
                    new_word_attributes["spaces"] = sentence_content.count(" ")

                is_ending_punctuation: bool = (word_index == len(sentence) - 1) and \
                                              (current_word in PSE_ENDING_PUNCTUATION)
                if current_parallelism_id is not None and is_ending_punctuation is False:
                    new_word_attributes["parallelism_id"] = current_parallelism_id
                    new_word_attributes["branch_id"] = branch_tracker[current_parallelism_id]

                new_word: PSEWord = PSEWord(**new_word_attributes)
                paibi_words.append(new_word)
        else:
            raise NotImplementedError(f"Currently, this code does not handle the tag <{sentence.tag}>")


def combine_annotations(paibi_words: list[PSEWord], new_annotations: list[BratAnnotation],
                        parallelism_count: int, paibi_sentences: list[tuple[int, int]]):
    if new_annotations is not None:
        current_document: str = ""
        overall_word_index: int = 0
        annotation_index: int = 0
        current_paragraph_index: int = -1

        parallelism_tracker: dict[int, int] = {}
        branch_tracker: dict[int, list[int]] = {}

        current_annotation: Optional[BratAnnotation] = new_annotations.pop(0) \
            if len(new_annotations) > 0 else None

        while current_annotation is not None and overall_word_index < len(paibi_words):
            current_word: PSEWord = paibi_words[overall_word_index]

            # If we're starting a new paragraph, we need to incorporate the <Paragraph X> text.
            while current_paragraph_index < current_word.paragraph_index:
                if current_paragraph_index != -1:
                    current_document += "\n"

                current_document += PSE_PARAGRAPH_TEMPLATE.format(current_word.paragraph_index + 1)
                current_paragraph_index += 1

            # If we're starting a new sentence, we need to count the newline
            #   and possibly the ">>> " used to make lines to-be-annotated visible.
            if current_word.word_index == 0:
                current_document += "\n"

                if len(paibi_sentences) > 0:
                    current_location: tuple[int, int] = (current_word.paragraph_index, current_word.sentence_index)
                    next_location: tuple[int, int] = paibi_sentences[0]
                    if current_location == next_location:
                        current_document += PSE_DESIGNATED_LINE_BEGINNING
                        paibi_sentences.pop(0)

            for _ in range(0, current_word.spaces):
                current_document += " "

            # We add an annotation if it is warranted.
            if current_annotation.start <= len(current_document) < current_annotation.end:
                add_annotation(
                    current_annotation, annotation_index, current_word, parallelism_count,
                    parallelism_tracker, branch_tracker
                )

            # We increment to the next word.
            current_document += current_word.word
            overall_word_index += 1

            # We check whether we are done with the current annotation.
            if len(current_document) >= current_annotation.end:
                annotation_index += 1
                if len(new_annotations) > 0:
                    current_annotation = new_annotations.pop(0)
                else:
                    current_annotation = None


def add_annotation(current_annotation: BratAnnotation, annotation_index: int, current_word: PSEWord,
                   parallelism_count: int, parallelism_tracker: dict[int, int],
                   branch_tracker: dict[int, list[int]]):
    if parallelism_tracker.get(current_annotation.parallelism_id, None) is None:
        parallelism_tracker[current_annotation.parallelism_id] = parallelism_count + len(parallelism_tracker)
        branch_tracker[current_annotation.parallelism_id] = [annotation_index]

    current_parallelism_id: int = parallelism_tracker[current_annotation.parallelism_id]

    if annotation_index not in branch_tracker[current_annotation.parallelism_id]:
        branch_tracker[current_annotation.parallelism_id].append(annotation_index)

    current_branch_id: int = branch_tracker[current_annotation.parallelism_id].index(annotation_index) + 1

    if current_word.parallelism_id is None and current_word.branch_id is None:
        current_word.parallelism_id = current_parallelism_id
        current_word.branch_id = current_branch_id
    else:
        current_word.alternate_parallelism_id = current_parallelism_id
        current_word.alternate_branch_id = current_branch_id


def write_combined_data(document_attributes: dict[str, str], scrawl: Optional[Element],
                        revised_words: list[PSEWord], output_filepath: str):
    del document_attributes["parallelisms"]
    root: Element = Element("doc", attrib=document_attributes)
    if scrawl is not None:
        root.append(scrawl)

    current_paragraph_index: Optional[int] = None
    current_sentence_index: Optional[int] = None
    for word_index, word in enumerate(revised_words):
        word_paragraph_index = word.paragraph_index
        if current_paragraph_index != word_paragraph_index:
            paragraph_attrib: dict[str, str] = {"id": str(word_paragraph_index)}
            new_paragraph: Element = Element("para", attrib=paragraph_attrib)
            root.append(new_paragraph)
            current_paragraph_index = word_paragraph_index
            current_sentence_index = None

        word_sentence_index = word.sentence_index
        if current_sentence_index != word_sentence_index:
            sentence_attrib: dict[str, str] = {"id": str(word_sentence_index)}
            new_sentence: Element = Element("sent", attrib=sentence_attrib)
            root[-1].append(new_sentence)
            current_sentence_index = word_sentence_index

        word_attrib: dict[str, str] = {
            "id": str(word.word_index),
            "cont": word.word,
            "pos": word.part_of_speech,
            "ne": word.named_entity,
            "parent": word.dependency_parent,
            "relate": word.dependency_tag,
        }

        if word.parallelism_id is not None and word.branch_id is not None:
            is_ending_punctuation: bool = (word.word in PSE_ENDING_PUNCTUATION) and \
                                          ((word_index == len(revised_words) - 1) or
                                           (revised_words[word_index + 1].parallelism_id != word.parallelism_id) or
                                           (revised_words[word_index + 1].branch_id != word.branch_id))

            if is_ending_punctuation is False:
                word_attrib["parallelism_id"] = str(word.parallelism_id)
                word_attrib["branch_id"] = str(word.branch_id)

        if word.alternate_parallelism_id is not None and word.alternate_branch_id is not None:
            word_attrib["alt_parallelism_id"] = str(word.alternate_parallelism_id)
            word_attrib["alt_branch_id"] = str(word.alternate_branch_id)

        word_element: Element = Element("word", attrib=word_attrib)
        root[-1][-1].append(word_element)

    output_tree: ElementTree = ElementTree(root)
    ETree.indent(output_tree, space="\t", level=0)
    output_tree.write(output_filepath, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--annotation-directory", type=str, default="data/pse/annotations", help=PSEMessage.ANNOTATION_DIRECTORY
    )
    parser.add_argument(
        "--output-directory", type=str, default="data/pse/gold/augmented", help=PSEMessage.OUTPUT_DIRECTORY
    )
    parser.add_argument("--pse-directory", type=str, default="data/pse/gold/original", help=PSEMessage.PSE_DIRECTORY)
    args: Namespace = parser.parse_args()

    for filepath in (args.pse_directory, args.annotation_directory, args.output_directory):
        if not path.isdir(filepath):
            raise ValueError(f"The filepath <{filepath}> is not recognized as a valid, existing directory.")

    # Step 1: coordinate filepaths in each directory together.
    filepaths: list[PSEFileGroup] = \
        coordinate_files(args.pse_directory, args.annotation_directory, args.output_directory)

    for (pse_path, annotation_pair_paths, output_path) in filepaths:
        # Step 2a: read in data from the original data source.
        pse_attributes, scrawl_elements, words, revised_sentences = read_paibi_file(pse_path)

        # Step 2b: read in data from the annotation directory.
        if annotation_pair_paths is not None:
            _, annotation_path = annotation_pair_paths
            brat_annotations: Optional[list[BratAnnotation]] = process_annotation_file(annotation_path)
            assert len(brat_annotations) > 0
        else:
            brat_annotations = None

        # Step 2c: combine the data read in from the two files.
        combine_annotations(words, brat_annotations, pse_attributes["parallelisms"], revised_sentences)

        # Step 2d: write the data to one output file, combining all the results into one format.
        write_combined_data(pse_attributes, scrawl_elements, words, output_path)
