from typing import Any, Union
from xml.etree import ElementTree as ETree
from xml.etree.ElementTree import ElementTree, Element

from utils.data.loaders.constants import CollectionFormat, DocumentState, EXCEPTIONS, UnitCollection
from utils.data.loaders.loader import BaseTagLoader
from utils.data.tags import TagLink


class SermonTagLoader(BaseTagLoader):
    def load_data(self, input_filepath: str, loading_kwargs: dict[str, Any]) -> tuple[dict[str, list], dict[str, Any]]:
        tree: ElementTree = ETree.parse(input_filepath)
        root: Element = tree.getroot()

        document_state: DocumentState = self._define_document_state(root)

        document_text: str = " ".join(root.itertext()).lower()
        structures: dict[str, list] = {
            "tokens": loading_kwargs["tokenizer"].tokenize(document_text, enclitics_exceptions=EXCEPTIONS),
            "tags": [[] for _ in range(0, loading_kwargs["tagging_kwargs"]["stratum_count"])]
        }

        for stratum in range(0, loading_kwargs["tagging_kwargs"]["stratum_count"]):
            document_state["current_stratum"] = stratum
            for section in root:
                self._read_element(section, structures, document_state, loading_kwargs)

            # We verify that the conversion has been done appropriately.
            assert len(structures["tokens"]) == len(structures["tags"][stratum])
            document_state["current_tokens"].clear()

        loaded_kwargs: dict[str, Any] = {"root": root}
        return structures, loaded_kwargs

    def unitize_data(self, structures: dict[str, list], loading_kwargs: dict[str, Any],
                     **loaded_kwargs) -> UnitCollection:
        units: UnitCollection = []

        root: Element = loaded_kwargs["root"]
        partition_id: int = 1

        if len(structures["tokens"]) > 0:
            if loading_kwargs["collection_format"] == CollectionFormat.DOCUMENT:
                self._collect_document_unit(units, structures, loading_kwargs)
            elif loading_kwargs["collection_format"] == CollectionFormat.SECTION:
                section_start: int = 0
                section_end: int = 0
                for section in root:
                    section_text: str = " ".join(section.itertext()).lower()
                    tokens: list[str] = loading_kwargs["tokenizer"].\
                        tokenize(section_text, enclitics_exceptions=EXCEPTIONS)
                    section_end += len(tokens)

                    self._collect_section_unit(
                        units, structures, loading_kwargs, section_start, section_end, partition_id
                    )

                    section_start = section_end
                    partition_id += 1
            else:
                raise ValueError(f"The collection format <{loading_kwargs['collection_format']}> "
                                 f"is not supported for this dataset.")
        else:
            raise ValueError("No tokens were produced in the processing of a file.")

        return units

    def _define_document_state(self, root: Element):
        back_pointers: dict[Element, Element] = {child: parent for parent in root.iter() for child in parent}
        stratification: dict[int, int] = self._stratify_parallelisms(root, back_pointers)

        branch_totals: dict[int, int] = self._count_branches(root)
        branch_tracker: dict[str, dict[int, int]] = {
            "current": {key: 0 for key in branch_totals.keys()},
            "total": branch_totals
        }

        distance_tracker: dict[int, list[int]] = {parallelism_id: [] for parallelism_id in stratification.keys()}

        document_state: DocumentState = {
            "back_pointers": back_pointers,
            "branch_tracker": branch_tracker,
            "current_tokens": [],
            "distance_tracker": distance_tracker,
            "stratification": stratification,
        }
        return document_state

    def _read_element(self, element: Element, structures: dict[str, list], state: DocumentState,
                      loading_kwargs: dict[str, Any]):
        tags: list[str] = self._generate_tagged_text(element, "text", state, structures["tokens"], loading_kwargs)
        structures["tags"][state["current_stratum"]].extend(tags)

        for child in element:
            self._read_element(child, structures, state, loading_kwargs)

        if element.tail is not None and element.tail.strip() != "":
            tags: list[str] = self._generate_tagged_text(element, "tail", state, structures["tokens"], loading_kwargs)
            structures["tags"][state["current_stratum"]].extend(tags)

    def _generate_tagged_text(self, element: Element, attribute: str, state: DocumentState, document_tokens: list[str],
                              loading_kwargs: dict[str, Any]) -> list[str]:
        is_alternate_stratum: bool = element.tag == "parallelism" and \
                                     state["stratification"][int(element.attrib["id"])] != state["current_stratum"]

        if (element.tag == "section" and attribute == "text") or \
                (element.tag == "parallelism" and attribute == "tail") or \
                is_alternate_stratum is True:

            # If the alternate stratum is a higher stratum, and we're currently handling the lower stratum,
            #   then we want to avoid double-tagging a truly nested branch.
            #   As a result, when we get to its XML node, we skip it in the tree.
            if is_alternate_stratum is True and self._is_covered_stratum(element, state) is True or \
                    getattr(element, attribute) is None:
                tags: list[str] = []
            else:
                text_segment: str = getattr(element, attribute).lower()
                tokens: list[str] = loading_kwargs["tokenizer"].\
                    tokenize(text_segment, enclitics_exceptions=EXCEPTIONS)
                self._check_tokens(document_tokens, state["current_tokens"], tokens)
                state["current_tokens"].extend(tokens)
                tags: list[str] = self._collect_nonbranch_tags(tokens, state, loading_kwargs["tagging_kwargs"])
        elif element.tag == "parallelism" and attribute == "text":
            parallelism_id: int = int(element.attrib["id"])
            text_segment: str = " ".join(element.itertext()).lower()
            tokens: list[str] = loading_kwargs["tokenizer"].tokenize(text_segment, enclitics_exceptions=EXCEPTIONS)
            self._check_tokens(document_tokens, state["current_tokens"], tokens)
            state["current_tokens"].extend(tokens)
            tags: list[str] = self._collect_branch_tags(tokens, parallelism_id, state, loading_kwargs["tagging_kwargs"])

            current_link: str = loading_kwargs['tagging_kwargs']['link']
            if current_link == TagLink.TOKEN_DISTANCE:
                incrementation_amount: int = len(tokens)
            elif current_link == TagLink.BRANCH_DISTANCE:
                incrementation_amount = 1
            else:
                raise NotImplementedError(f"The link <{current_link}> is not currently supported.")
            self._increment_distance_tracker(state, incrementation_amount, parallelism_id)
            state["branch_tracker"]["current"][parallelism_id] += 1
        elif element.tag == "section" and attribute == "tail":
            raise ValueError("Text is present in the tail of a section, which is not intended for this data.")
        else:
            raise ValueError(f"Unrecognized tag <{element.tag}>.")

        return tags

    @staticmethod
    def _count_branches(root: Element) -> dict[int, int]:
        branch_totals: dict[int, int] = {}
        for element in root.iter("parallelism"):
            parallelism_id: int = int(element.attrib["id"])
            if branch_totals.get(parallelism_id, None) is None:
                branch_totals[parallelism_id] = 0
            branch_totals[parallelism_id] += 1
        return branch_totals

    @staticmethod
    def _stratify_parallelisms(root: Element, back_pointers: dict[Element, Element]) -> dict[int, int]:
        stratified_parallelisms: dict[int, int] = {}
        for element in root.iter("parallelism"):
            parallelism_id: int = int(element.attrib["id"])

            # We determine the depth of the element to see what level of nesting on which it should occur.
            element_depth: int = 0
            ancestor: Element = back_pointers[element]
            while ancestor.tag not in ("section", "sermon"):
                element_depth += 1
                ancestor = back_pointers[ancestor]

            # Since any branch could cause the nesting of the parallelism to go up,
            # the level for a parallelism is equivalent to the most highly-nested branch.
            stratified_parallelisms[parallelism_id] = max(stratified_parallelisms.get(parallelism_id, 0), element_depth)

        return stratified_parallelisms

    @staticmethod
    def _is_any_parallelism_incomplete(state: DocumentState) -> bool:
        incomplete_bool: bool = False
        branch_tracker: dict[str, dict[int, int]] = state["branch_tracker"]
        for (parallelism_id, stratum) in state["stratification"].items():
            if stratum != state["current_stratum"]:
                continue
            elif 0 < branch_tracker["current"][parallelism_id] < branch_tracker["total"][parallelism_id]:
                incomplete_bool = True
                break
        return incomplete_bool

    @staticmethod
    def _is_covered_stratum(element: Element, state: DocumentState):
        covered_stratum_bool: bool = False
        if state["current_stratum"] < state["stratification"][int(element.attrib["id"])] and \
                state["back_pointers"][element].tag == "parallelism":
            covered_stratum_bool = True
        return covered_stratum_bool

    @staticmethod
    def _get_collection_id(filename: str, backup_id: int, loading_kwargs: dict[str, Any]) -> Union[int, str]:
        filename_split: list[str] = filename.split("_", maxsplit=1)
        if len(filename_split) > 1:
            # All sermon filenames contain a _, separating their ID from the rest of the text.
            collection_id, _ = filename_split
        else:
            collection_id = backup_id

        return collection_id

    @staticmethod
    def _check_tokens(document_tokens: list[str], current_tokens: list[str], tentative_tokens: list[str]):
        # This function attempts to heuristically fix tokenization issues
        #   that may occur between global and local contexts.
        # It assumes that these issues are due to oversegmentation,
        #   and it attempts to solve the oversegmentation problem with respect to the local context
        #   so that it matches the global context.

        current_token_length: int = len(current_tokens)
        document_token_sequence: list[str] = \
            document_tokens[current_token_length:current_token_length + len(tentative_tokens)]

        if document_token_sequence != tentative_tokens:
            index: int = 0
            change_count: int = 0
            while index < len(tentative_tokens):
                if index == len(tentative_tokens) - 1:
                    if tentative_tokens[:-1] == document_token_sequence[:(-1 + -1 * change_count)]:
                        break
                    else:
                        index += 1
                elif tentative_tokens[index] != document_token_sequence[index]:
                    change_count += 1

                    if index == 0:
                        current_tokens[-1] += tentative_tokens[index]
                        del tentative_tokens[index]
                    else:
                        tentative_tokens[index] = tentative_tokens[index] + tentative_tokens[index + 1]
                        del tentative_tokens[index + 1]

                    if tentative_tokens == document_token_sequence[:(-1 * change_count)]:
                        break
                    else:
                        index += 1
                else:
                    index += 1
            else:
                raise ValueError(f"The element sequence <{tentative_tokens}> is not equal to "
                                 f"the document sequence <{document_token_sequence}>.")
