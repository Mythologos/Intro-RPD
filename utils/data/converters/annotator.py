from re import match, Match
from typing import Optional

from utils.data.converters.constants import BratAnnotation


ANNOTATION_PARSING_REGEX: str = r"(?:(?P<text_annotation>" \
                                r"(?P<text_header>T(?P<text_header_number>[\d]+))[\s]*" \
                                r"(?P<text_label>[\w]+)[\s]*" \
                                r"(?P<start>[\d]+)(?:[\s]+[\d]+;[\d]+)*[\s]+(?P<end>[\d]+)[\s]*" \
                                r"(?P<text>.+)?)|" \
                                r"(?P<relation>" \
                                r"(?P<relation_header>R[\d]+)[\s]*" \
                                r"(?P<relation_label>[\w]+)[\s]*" \
                                r"(?:Arg[\d]+:T(?P<relation_arg1>[\d]+))[\s]*" \
                                r"(?:Arg[\d]+:T(?P<relation_arg2>[\d]+)))|" \
                                r"(?P<annotator_note>(?P<note_header>#[\d]+)[\s]*" \
                                r"(?P<note_label>[\w]+)[\s]*" \
                                r"(?P<related_entity>T[\d]+)[\s]*" \
                                r"(?P<note_text>.+)))"


def process_annotation_file(annotation_filepath: str) -> list[BratAnnotation]:
    annotation_queue: list[BratAnnotation] = []
    current_entities: list[Match] = []
    entity_used_table: list[bool] = []
    current_relations: list[Match] = []
    with open(annotation_filepath, mode="r", encoding="utf-8") as annotation_file:
        for line in annotation_file:
            annotation: Match = match(ANNOTATION_PARSING_REGEX, line)
            if not annotation:
                raise ValueError(f"Line <{line}> not recognized.")
            elif annotation["note_header"]:
                # We skip notes, since they're not incorporated into the data.
                continue
            elif annotation["text_header"] and annotation["text_header"].startswith("T"):
                current_entities.append(annotation)
                entity_used_table.append(False)
            elif annotation["relation_header"] and annotation["relation_header"].startswith("R"):
                current_relations.append(annotation)
            else:
                raise ValueError(f"Line produces annotation with unknown identification.\n"
                                 f"Line: <{line}>")

    # Once we have all the lines organized, we can use the relation annotations to categorize
    # and to group the entities together such that they can be tagged. In particular,
    # we load them into the BratAnnotations.

    current_entities.sort(key=lambda ent: int(ent["text_header_number"]))
    current_relations.sort(key=lambda rel: get_relation_key(rel, current_entities))

    # We keep a lookup table in order to chain together annotations once they are appended to the annotation_queue.
    annotation_lookup_table: list[Optional[int]] = [None for _ in range(0, len(current_entities))]

    # We also keep a parallelism ID indicating what the next one should be.
    new_parallelism_id: int = 1
    for relation in current_relations:
        # First, we need to determine which entity should go first. Earlier ones in the text should always go first.
        first_entity_index, second_entity_index = get_entity_indices(relation, current_entities)

        first, second = current_entities[first_entity_index], current_entities[second_entity_index]
        parallelism_id = new_parallelism_id
        if entity_used_table[first_entity_index] is False:
            annotation_queue.append(BratAnnotation(int(first["start"]), int(first["end"]), parallelism_id))
            entity_used_table[first_entity_index] = True
            annotation_lookup_table[first_entity_index] = len(annotation_queue) - 1
        else:
            parallelism_id = annotation_queue[annotation_lookup_table[first_entity_index]].parallelism_id

        if entity_used_table[second_entity_index] is False:
            annotation_queue.append(BratAnnotation(int(second["start"]), int(second["end"]), parallelism_id))
            entity_used_table[second_entity_index] = True
            annotation_lookup_table[second_entity_index] = len(annotation_queue) - 1

        if new_parallelism_id == parallelism_id:
            new_parallelism_id += 1

    # We sort by the start index first; then, we sort by larger end indexes;
    # then, we sort by the largest parallelism index.
    # We do so to balance the XML: the longest spans should be the most "outer",
    # and the shortest should be the most "inner".
    annotation_queue.sort(key=lambda ann: (ann.start, -1 * ann.end, -1 * ann.parallelism_id))
    return annotation_queue


def get_entity_indices(relation: Match, current_entities: list[Match]):
    first_arg_index: int = int(relation["relation_arg1"]) - 1
    second_arg_index: int = int(relation["relation_arg2"]) - 1
    first_arg, second_arg = current_entities[first_arg_index], current_entities[second_arg_index]
    if int(first_arg["start"]) < int(second_arg["start"]):
        first_entity_index: int = first_arg_index
        second_entity_index: int = second_arg_index
    elif int(second_arg["start"]) < int(first_arg["start"]):
        first_entity_index: int = second_arg_index
        second_entity_index: int = first_arg_index
    else:
        raise ValueError("Two branches start at the same point.")

    return first_entity_index, second_entity_index


def get_relation_key(relation: Match, entities: list[Match]) -> int:
    first_arg_index: int = int(relation["relation_arg1"]) - 1
    second_arg_index: int = int(relation["relation_arg2"]) - 1
    first_arg = entities[first_arg_index]
    second_arg = entities[second_arg_index]
    sorting_key = min(int(first_arg["start"]), int(second_arg["start"]))
    return sorting_key
