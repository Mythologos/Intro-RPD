from abc import abstractmethod
from collections import deque
from typing import Any, Optional, Sequence, Union
from xml.etree import ElementTree as ETree
from xml.etree.ElementTree import Element, ElementTree

from aenum import NamedConstant

from utils.data.loaders.constants import CollectionFormat, DocumentState, UnitCollection, NORMALIZATION_TABLE
from utils.data.loaders.loader import BaseTagLoader
from utils.data.tags import BIOTag, TagLink


class PaibiTag(NamedConstant):
    INNER_BEGINNING: str = "inParaPaibiB"
    INNER_MIDDLE: str = "inParaPaibiI"
    INNER_ENDING: str = "inParaPaibiE"
    OUTER_BEGINNING: str = "outParaPaibiB"
    OUTER_MIDDLE: str = "outParaPaibiI"
    OUTER_ENDING: str = "outParaPaibiE"
    SINGLE_SENTENCE: str = "InSentenceBIE"


PAIBI_INNER_TAGS: Sequence[str] = (PaibiTag.INNER_BEGINNING, PaibiTag.INNER_MIDDLE, PaibiTag.INNER_ENDING)
PAIBI_OUTER_TAGS: Sequence[str] = (PaibiTag.OUTER_BEGINNING, PaibiTag.OUTER_MIDDLE, PaibiTag.OUTER_ENDING)
PAIBI_BEGINNING_TAGS: Sequence[str] = (PaibiTag.INNER_BEGINNING, PaibiTag.OUTER_BEGINNING)
PAIBI_MIDDLE_TAGS: Sequence[str] = (PaibiTag.INNER_MIDDLE, PaibiTag.OUTER_MIDDLE)
PAIBI_ENDING_TAGS: Sequence[str] = (PaibiTag.INNER_ENDING, PaibiTag.OUTER_ENDING)


class BasePaibiTagLoader(BaseTagLoader):
    def load_data(self, input_filepath: str, loading_kwargs: dict[str, Any]) -> tuple[dict[str, list], dict[str, Any]]:
        tree: ElementTree = ETree.parse(input_filepath)
        root: Element = tree.getroot()

        document_state: DocumentState = self._define_document_state(root, loading_kwargs)

        translation_table: dict = str.maketrans(NORMALIZATION_TABLE)
        document_tokens: list[str] = \
            [word.attrib["cont"].translate(translation_table) for word in root.findall(".//word")]
        structures: dict[str, list] = {
            "tokens": document_tokens,
            "tags": [[] for _ in range(0, loading_kwargs["tagging_kwargs"]["stratum_count"])]
        }

        for stratum in range(0, loading_kwargs["tagging_kwargs"]["stratum_count"]):
            # We know a priori that this dataset is flat in terms of its parallel structure.
            #   So, we expedite the collection of multiple strata by only iterating over the document when necessary.
            document_state["current_stratum"] = stratum
            if stratum == 0:
                self._read_document(root, structures, document_state, loading_kwargs)
            else:
                structures["tags"][stratum].extend([BIOTag.OUTSIDE.value for _ in range(0, len(document_tokens))])

            # We verify that the conversion has been done appropriately.
            assert len(structures["tokens"]) == len(structures["tags"][stratum])

        loaded_kwargs: dict[str, Any] = {"root": root}
        return structures, loaded_kwargs

    def unitize_data(self, structures: dict[str, list], loading_kwargs: dict[str, Any], **loaded_kwargs):
        units: UnitCollection = []

        if len(structures["tokens"]) > 0:
            if loading_kwargs["collection_format"] == CollectionFormat.DOCUMENT:
                self._collect_document_unit(units, structures, loading_kwargs)
            elif loading_kwargs["collection_format"] == CollectionFormat.SECTION:
                partition_id: int = 1
                root: Element = loaded_kwargs["root"]
                paragraph_start, paragraph_end = 0, 0

                for paragraph in root.iter("para"):
                    paragraph_tokens: list[str] = []
                    for sentence in paragraph.iter("sent"):
                        sentence_tokens: list[str] = [word.attrib["cont"] for word in sentence.iter("word")]
                        paragraph_tokens.extend(sentence_tokens)
                    paragraph_end += len(paragraph_tokens)

                    if paragraph_start == paragraph_end:
                        continue
                    else:
                        self._collect_section_unit(
                            units, structures, loading_kwargs, paragraph_start, paragraph_end, partition_id
                        )

                        paragraph_start = paragraph_end
                        partition_id += 1
        else:
            raise ValueError("No tokens were produced in the processing of a file.")

        return units

    @staticmethod
    def _is_any_parallelism_incomplete(state: DocumentState):
        incomplete_bool: bool = False
        branch_tracker: dict[str, dict[int, int]] = state["branch_tracker"]
        for parallelism_id in branch_tracker["current"].keys():
            if 0 < branch_tracker["current"][parallelism_id] < branch_tracker["total"][parallelism_id]:
                incomplete_bool = True
                break
        return incomplete_bool

    @staticmethod
    def _get_collection_id(filename: str, backup_id: int, loading_kwargs: dict[str, Any]) -> Union[int, str]:
        collection_id: Union[str, int] = filename if filename != "" else backup_id
        return collection_id

    @staticmethod
    def _get_link_incrementation(tokens: list[str], loading_kwargs: dict[str, Any]) -> int:
        current_link: str = loading_kwargs['tagging_kwargs']['link']
        if current_link == TagLink.TOKEN_DISTANCE:
            incrementation_amount: int = len(tokens)
        elif current_link == TagLink.BRANCH_DISTANCE:
            incrementation_amount = 1
        else:
            raise NotImplementedError(f"The link <{current_link}> is not currently supported.")
        return incrementation_amount

    @abstractmethod
    def _define_document_state(self, root: Element, loading_kwargs: dict[str, Any]) -> DocumentState:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _count_branches(root: Element, loading_kwargs: dict[str, Any]) -> dict[int, int]:
        raise NotImplementedError

    @abstractmethod
    def _read_document(self, root: Element, structures: dict[str, list], state: DocumentState,
                       loading_kwargs: dict[str, Any]):
        raise NotImplementedError


class OriginalPaibiTagLoader(BasePaibiTagLoader):
    def _define_document_state(self, root: Element, loading_kwargs: dict[str, Any]) -> DocumentState:
        branch_totals: dict[int, int] = self._count_branches(root, loading_kwargs)
        branch_tracker: dict[str, dict[int, int]] = {"current": {}, "total": branch_totals}
        distance_tracker: dict[int, list[int]] = {parallelism_id: [] for parallelism_id in branch_totals.keys()}

        # We know a priori that the dataset is flat, so everything is on the same layer.
        stratification: dict[int, int] = {parallelism_id: 0 for parallelism_id in branch_totals.keys()}

        document_state: DocumentState = {
            "branch_tracker": branch_tracker,
            "distance_tracker": distance_tracker,
            "inner_queue": deque([]),
            "outer_queue": deque([]),
            "stratification": stratification
        }
        return document_state

    def _read_document(self, root: Element, structures: dict[str, list], state: DocumentState,
                       loading_kwargs: dict[str, Any]):
        collection_bool: bool = loading_kwargs["collection_format"] == CollectionFormat.SECTION
        tagging_kwargs: dict[str, str] = loading_kwargs["tagging_kwargs"]
        for sentence in root.iter("sent"):
            tokens: list[str] = [word.attrib["cont"] for word in sentence.iter("word")]
            sentence_type: Optional[str] = sentence.attrib.get("type", None)
            if sentence_type in PaibiTag:   # The sentence represents a set of inside tags.
                if sentence_type == PaibiTag.SINGLE_SENTENCE or \
                        (sentence_type in PAIBI_OUTER_TAGS and collection_bool is True):
                    tags: list[str] = self._collect_nonbranch_tags(tokens, state, tagging_kwargs)
                else:
                    parallelism_id: int = self._retrieve_parallelism_id(sentence, state)
                    tags: list[str] = self._collect_branch_tags(tokens, parallelism_id, state, tagging_kwargs)
                    incrementation_amount: int = self._get_link_incrementation(tokens, loading_kwargs)
                    self._increment_distance_tracker(state, incrementation_amount, parallelism_id)
                    state["branch_tracker"]["current"][parallelism_id] += 1
            else:   # The sentence represents a set of outside tags.
                tags: list[str] = self._collect_nonbranch_tags(tokens, state, tagging_kwargs)

            structures["tags"][state["current_stratum"]].extend(tags)

    @staticmethod
    def _count_branches(root: Element, loading_kwargs: dict[str, Any]) -> dict[int, int]:
        collection_bool: bool = (loading_kwargs["collection_format"] == CollectionFormat.SECTION)
        branch_totals: dict[int, int] = {}

        inner_queue: deque[int] = deque([])
        outer_queue: deque[int] = deque([])
        for element in root.iter("sent"):
            element_type: Optional[str] = element.attrib.get("type", None)
            if element_type in PAIBI_INNER_TAGS or element_type in PAIBI_OUTER_TAGS:
                if element_type in PAIBI_INNER_TAGS:
                    relevant_stack = inner_queue
                else:   # element_type in OUTER_TAGS:
                    if collection_bool:
                        continue
                    else:
                        relevant_stack = outer_queue

                if element_type in PAIBI_BEGINNING_TAGS:
                    relevant_stack.appendleft(len(branch_totals))
                    branch_totals[relevant_stack[0]] = 1
                elif element_type in PAIBI_MIDDLE_TAGS:
                    branch_totals[relevant_stack[0]] += 1
                else:   # element_type in PAIBI_ENDING_TAGS:
                    branch_totals[relevant_stack[0]] += 1
                    relevant_stack.popleft()
            elif element_type == PaibiTag.SINGLE_SENTENCE:
                pass
            elif element_type not in (" ", None):
                raise ValueError(f"The tag type <{element_type}> is not recognized.")

        assert (len(inner_queue) == 0) and (len(outer_queue) == 0)
        return branch_totals

    @staticmethod
    def _retrieve_parallelism_id(sentence: Element, state: DocumentState) -> int:
        sentence_type: str = sentence.attrib["type"]
        if sentence_type in PAIBI_INNER_TAGS:
            relevant_stack: deque = state["inner_queue"]
        else:    # sentence_type in OUTER_TAGS:
            relevant_stack: deque = state["outer_queue"]

        if sentence_type in PAIBI_BEGINNING_TAGS:
            parallelism_id: int = len(state["branch_tracker"]["current"])
            state["branch_tracker"]["current"][parallelism_id] = 0
            relevant_stack.appendleft(parallelism_id)
        elif sentence_type in PAIBI_MIDDLE_TAGS:
            parallelism_id: int = relevant_stack[0]
        else:   # sentence_type in PAIBI_ENDING_TAGS:
            parallelism_id: int = relevant_stack.popleft()

        return parallelism_id


class AugmentedPaibiTagLoader(BasePaibiTagLoader):
    def _define_document_state(self, root: Element, loading_kwargs: dict[str, Any]) -> DocumentState:
        branch_totals: dict[int, int] = self._count_branches(root, loading_kwargs)
        branch_tracker: dict[str, dict[int, int]] = {"current": {}, "total": branch_totals}
        distance_tracker: dict[int, list[int]] = {parallelism_id: [] for parallelism_id in branch_totals.keys()}

        # We know a priori that the dataset is flat, so everything is on the same layer.
        stratification: dict[int, int] = {parallelism_id: 0 for parallelism_id in branch_totals.keys()}

        document_state: DocumentState = {
            "branch_tracker": branch_tracker,
            "distance_tracker": distance_tracker,
            "inner_queue": deque([]),
            "stratification": stratification
        }
        return document_state

    def _read_document(self, root: Element, structures: dict[str, list], state: DocumentState,
                       loading_kwargs: dict[str, Any]):
        tagging_kwargs: dict[str, str] = loading_kwargs["tagging_kwargs"]

        current_tokens: list[str] = []
        current_word_index: int = 0

        words: list[Element] = list(root.iter("word"))
        while current_word_index < len(words):
            current_word: Element = words[current_word_index]
            current_tokens.append(current_word.attrib["cont"])

            next_word_index: int = current_word_index + 1
            if current_word.attrib.get("parallelism_id_1", None) is not None:
                current_parallelism_id: int = int(current_word.attrib["parallelism_id_1"])
                current_branch_id: int = int(current_word.attrib["branch_id_1"])
                current_branch_identifier = (current_parallelism_id, current_branch_id)

                if current_parallelism_id not in state["branch_tracker"]["current"]:
                    state["branch_tracker"]["current"][current_parallelism_id] = 0

                while next_word_index < len(words):
                    next_word: Element = words[next_word_index]
                    if next_word.attrib.get("parallelism_id_1", None) is not None:
                        next_parallelism_id: int = int(next_word.attrib["parallelism_id_1"])
                        next_branch_id: int = int(next_word.attrib["branch_id_1"])
                        next_branch_identifier = (next_parallelism_id, next_branch_id)
                        if current_branch_identifier == next_branch_identifier:
                            current_tokens.append(next_word.attrib["cont"])
                            next_word_index += 1
                        else:
                            break
                    else:
                        break

                new_tags: list[str] = \
                    self._collect_branch_tags(current_tokens, current_parallelism_id, state, tagging_kwargs)
                incrementation_amount: int = self._get_link_incrementation(current_tokens, loading_kwargs)
                self._increment_distance_tracker(state, incrementation_amount, current_parallelism_id)
                state["branch_tracker"]["current"][current_parallelism_id] += 1
            else:
                while next_word_index < len(words):
                    next_word: Element = words[next_word_index]
                    if next_word.attrib.get("parallelism_id_1", None) is not None:
                        break
                    else:
                        current_tokens.append(next_word.attrib["cont"])
                        next_word_index += 1

                new_tags: list[str] = self._collect_nonbranch_tags(current_tokens, state, tagging_kwargs)

            structures["tags"][state["current_stratum"]].extend(new_tags)
            current_tokens = []
            current_word_index = next_word_index

    @staticmethod
    def _count_branches(root: Element, loading_kwargs: dict[str, Any]) -> dict[int, int]:
        branch_totals: dict[int, int] = {}
        branch_identifiers: set[tuple[int, int]] = set()

        for word in root.iter("word"):
            if (word.attrib.get("parallelism_id_1", None) is not None and
                    word.attrib.get("branch_id_1", None) is not None):
                parallelism_id: int = int(word.attrib["parallelism_id_1"])
                branch_id: int = int(word.attrib["branch_id_1"])
                branch_identifier: tuple[int, int] = (parallelism_id, branch_id)
                if branch_identifier not in branch_identifiers:
                    if parallelism_id not in branch_totals:
                        branch_totals[parallelism_id] = 0
                    branch_totals[parallelism_id] += 1
                    branch_identifiers.add(branch_identifier)

        return branch_totals
