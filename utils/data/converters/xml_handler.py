import heapq

from re import Match, search, sub
from string import punctuation
from typing import Any, Optional, TextIO
from xml.etree.ElementTree import Element, ElementTree

from utils.data.converters.constants import BratAnnotation, FlagDict, SERMON_TITLE_REGEX

END_PUNCTUATION_REGEX: str = r"(?P<ending_punctuation>[!\"#$%&'()*+,-./:;<=>?@\[\]^_`{|}~]+)$"


def produce_xml(text_filepath: str, annotation_queue: list[BratAnnotation], output_filepath: str, flags: FlagDict):
    structures: dict[str, Any] = {
        "annotation_queue": annotation_queue,
        "back_pointers": {},
        "element_state": {"element": None, "attribute": "text"},
        "id_tracker": {},
        "pending_heap": []
    }

    # We open the text file and begin tracking characters on it.
    # For each word, we add the length and check the index.
    # Since the annotations are ordered, we don't need to iterate over them:
    # we just need to check the current character index against the first one.

    text_file: TextIO = open(text_filepath, mode="r", encoding="utf-8")
    first_sermon_header: str = text_file.readline()
    parsed_sermon_header: Match = search(SERMON_TITLE_REGEX, first_sermon_header)
    text_file.seek(0)  # We go back to the start, as we "peeked" at the first line for labeling purposes.
    lines: list[str] = text_file.readlines()
    text_file.close()

    root: Element = Element("sermon", attrib={"id": parsed_sermon_header["primary_id"]})
    tree: ElementTree = ElementTree(root)

    overall_index: int = 0
    for line_index, line in enumerate(lines):
        title: Optional[Match] = search(SERMON_TITLE_REGEX, line)
        if title is None:
            # If we don't hit a new section, then we handle the line that we're given.
            overall_index = append_line(structures, flags, line, overall_index)
        else:
            # If we hit a new section, then we'll need to hop out of any tree structures we've been building.
            # We'll want to add a new element to the root which is a new section.
            overall_index = handle_section_boundary(tree, structures, flags, title, line, overall_index)

    if flags["punctuation_strategy"] == "exclude":
        exclude_xml_punctuation(root, structures, flags)

    # We write out the contents of the XML tree.
    tree.write(output_filepath, encoding="utf-8", xml_declaration=True)


def handle_section_boundary(tree: ElementTree, structures: dict[str, Any], flags: FlagDict,
                            title: Match, line: str, overall_index: int) -> int:
    root: Element = tree.getroot()

    if flags["sectioning"] is True and title is not None:
        new_section: Element = Element("section", attrib={"id": title["secondary_id"]})
        root.append(new_section)

        structures["back_pointers"][new_section] = root
        structures["element_state"]["element"] = new_section
        structures["element_state"]["attribute"] = "text"

    # Before we conclude, we trim tags that are fully passed.
    overall_index += len(line)
    while len(structures["pending_heap"]) > 0 and overall_index >= structures["pending_heap"][0]:
        heapq.heappop(structures["pending_heap"])

    return overall_index


def append_line(structures: dict[str, Any], flags: FlagDict, line: str, overall_index: int) -> int:
    annotation_queue: list[BratAnnotation] = structures["annotation_queue"]
    pending_heap: list[int] = structures["pending_heap"]

    line_index: int = 0
    current_segment: str = ""

    while line_index < len(line):
        # We check if we are entering or leaving a new node.
        starting_annotation: bool = (annotation_queue and overall_index + line_index >= annotation_queue[0].start)
        finishing_annotation: bool = (pending_heap and overall_index + line_index >= pending_heap[0])
        if starting_annotation or finishing_annotation:
            current_segment = process_text_segment(current_segment, flags)
            write_segment(structures, current_segment)
            current_segment = ""

        handle_closing_tags(structures, overall_index, line_index)
        handle_opening_tags(structures, overall_index, line_index)

        # We add the next character to the current segment.
        current_segment += line[line_index]
        line_index += 1
    else:
        # We write what remains for the line, if anything.
        current_segment = process_text_segment(current_segment, flags)
        write_segment(structures, current_segment)

    overall_index += len(line)
    return overall_index


def handle_closing_tags(structures: dict[str, Any], overall_index: int, line_index: int):
    # If any tags are closing, we get rid of them from the pending heap.
    # If more than one tag closes, then we must move back up the tree; otherwise, we use the tail of the current node.
    while len(structures["pending_heap"]) > 0 and overall_index + line_index >= structures["pending_heap"][0]:
        heapq.heappop(structures["pending_heap"])

        # We proceed back up an element in the tree.
        if structures["element_state"]["attribute"] != "tail":
            structures["element_state"]["attribute"] = "tail"
        else:   # structures["element_state"]["attribute"] == "tail":
            current_element: Element = structures["element_state"]["element"]
            structures["element_state"]["element"] = structures["back_pointers"][current_element]


def handle_opening_tags(structures: dict[str, Any], overall_index: int, line_index: int):
    # We check whether there are any tags that are opening.
    while len(structures["annotation_queue"]) > 0 and \
            overall_index + line_index >= structures["annotation_queue"][0].start:
        # We get the current annotation...
        opening_annotation: BratAnnotation = structures["annotation_queue"].pop(0)

        # We update the ID Tracker ...
        if structures["id_tracker"].get(opening_annotation.parallelism_id, None) is None:
            structures["id_tracker"][opening_annotation.parallelism_id] = 0
        structures["id_tracker"][opening_annotation.parallelism_id] += 1

        # We create the new element ...
        new_parallelism_attributes: dict = {
            "id": str(opening_annotation.parallelism_id),
            "part": str(structures["id_tracker"][opening_annotation.parallelism_id])
        }
        new_parallelism: Element = Element("parallelism", attrib=new_parallelism_attributes)

        if structures["element_state"]["attribute"] == "text":
            # If we're in the text of an element, then we know we're nesting.
            structures["element_state"]["element"].append(new_parallelism)
            structures["back_pointers"][new_parallelism] = structures["element_state"]["element"]
        else:   # structures["element_state"]["attribute"] == "tail"
            # If we're in the tail of an element, then we're a sibling.
            parent_element: Element = structures["back_pointers"][structures["element_state"]["element"]]
            parent_element.append(new_parallelism)
            structures["back_pointers"][new_parallelism] = parent_element
            structures["element_state"]["attribute"] = "text"

        structures["element_state"]["element"] = new_parallelism

        # We place the current annotation, which we are now "inside", in the pending heap.
        heapq.heappush(structures["pending_heap"], opening_annotation.end)


def write_segment(structures: dict[str, Any], segment: str):
    segment = segment.replace("\n", " ")
    current_element: Element = structures["element_state"]["element"]
    current_text: Optional[str] = getattr(current_element, structures["element_state"]["attribute"])
    if current_text is None:
        setattr(current_element, structures["element_state"]["attribute"], segment)
    else:
        setattr(current_element, structures["element_state"]["attribute"], current_text + segment)


def exclude_xml_punctuation(element: Element, structures: dict[str, Any], flags: FlagDict):
    for child in element:
        if child.tag == "parallelism":
            # We first handle the text of the child...
            if child.text is not None:
                punctuated_text_match: Optional[Match] = search(END_PUNCTUATION_REGEX, child.text)
                if punctuated_text_match is not None:
                    child.text = sub(END_PUNCTUATION_REGEX, "", child.text)
                    ending_punctuation: str = punctuated_text_match.group("ending_punctuation")
                    # It's possible that the punctuation,
                    # having been bumped down to the tail, is still at an end.
                    if child.tail is None:
                        child.tail = ending_punctuation
                    else:
                        child.tail = ending_punctuation + child.tail

            # Then, we handle its children ...
            exclude_xml_punctuation(child, structures, flags)

            # Finally, we handle its tail ...
            parent: Element = structures["back_pointers"][child]
            if child.tail is not None and parent.tag == "parallelism":
                punctuated_tail_match: Optional[Match] = search(END_PUNCTUATION_REGEX, child.tail)
                if punctuated_tail_match is not None:
                    child.tail = sub(END_PUNCTUATION_REGEX, "", child.tail)
                    ending_punctuation: str = punctuated_tail_match.group("ending_punctuation")
                    if parent.tail is None:
                        parent.tail = ending_punctuation
                    else:
                        parent.tail = ending_punctuation + parent.tail
        else:
            exclude_xml_punctuation(child, structures, flags)


def process_text_segment(text: str, flags: FlagDict) -> str:
    revised_text: str = text
    if flags["punctuation"] is False:
        for punctuation_character in punctuation:
            revised_text = revised_text.replace(punctuation_character, "")

    if flags["capitalization"] is False:
        revised_text = revised_text.lower()

    if flags["format"] == "xml":
        for special_character in list(XMLSpecialCharacter):   # type: ignore
            revised_text = revised_text.replace(special_character.default, special_character.xml)

    return revised_text
