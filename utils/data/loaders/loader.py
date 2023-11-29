from abc import abstractmethod
from typing import Any, Optional, Union

from utils.data.loaders.constants import DocumentState, TagUnit, UnitCollection, UnitIdentifier
from utils.data.tags import BIOTag, TagLink, END_TAGSETS, MIDDLE_TAGSETS, NONINITIAL_TAGSETS


class BaseTagLoader:
    def __call__(self, directory: str, filename: str, backup_id: int, loading_kwargs: dict[str, Any]) -> UnitCollection:
        loading_kwargs["collection_id"] = self._get_collection_id(filename, backup_id, loading_kwargs)
        filepath: str = f"{directory}/{filename}"
        loaded_data, loaded_kwargs = self.load_data(filepath, loading_kwargs)
        unitized_data: UnitCollection = self.unitize_data(loaded_data, loading_kwargs, **loaded_kwargs)
        return unitized_data

    @staticmethod
    def _increment_distance_tracker(state: DocumentState, amount: int, current_parallelism: Optional[int] = None):
        for parallelism_id, distances in state["distance_tracker"].items():
            if state["stratification"][parallelism_id] != state["current_stratum"]:
                continue
            elif current_parallelism is not None and parallelism_id == current_parallelism:
                distances.append(1)
            elif len(distances) > 0:
                distances[-1] += amount

    @staticmethod
    def _collect_branch_tags(tokens: list[str], parallelism_id: int, state: DocumentState,
                             tagging_kwargs: dict[str, str]) -> list[str]:
        # If we're in a parallelism, then the tag (for the relevant stratum) must be a B, B-X, I, J, or E.
        tags: list[str] = []

        branch_tracker: dict[str, dict[int, int]] = state["branch_tracker"]
        for i in range(0, len(tokens)):
            if i == 0:
                # In this case, we're at the beginning of the branch. So, our options are B or B-X.
                tag: str = BIOTag.INITIAL_BEGINNING.value
                if branch_tracker["current"][parallelism_id] > 0:
                    if tagging_kwargs["link"] in (TagLink.TOKEN_DISTANCE, TagLink.BRANCH_DISTANCE):
                        distance_link: int = state["distance_tracker"][parallelism_id][-1]
                    else:
                        raise ValueError(f"The tag link <{tagging_kwargs['link']}> is currently not supported.")
                    tag = f"{tag}-{distance_link}"
            else:
                # In this case, we're in the middle or at the end of a branch. So, our options are I, J, or E.
                if tagging_kwargs["tagset"] in END_TAGSETS and i == (len(tokens) - 1):
                    tag: str = BIOTag.END.value
                elif tagging_kwargs["tagset"] in NONINITIAL_TAGSETS and branch_tracker["current"][parallelism_id] != 0:
                    tag: str = BIOTag.NONINITIAL_INSIDE.value
                else:
                    tag: str = BIOTag.INITIAL_INSIDE.value
            tags.append(tag)

        return tags

    def _collect_nonbranch_tags(self, tokens: list[str], state: DocumentState,
                                tagging_kwargs: dict[str, str]) -> list[str]:
        # From this, we know that the tag must be O or M.
        if self._is_any_parallelism_incomplete(state) is True and tagging_kwargs["tagset"] in MIDDLE_TAGSETS:
            specified_tag: str = BIOTag.MIDDLE.value
        else:
            specified_tag = BIOTag.OUTSIDE.value
        tags: list[str] = [specified_tag for _ in range(0, len(tokens))]

        if tagging_kwargs["link"] == TagLink.TOKEN_DISTANCE:
            incrementation_amount: int = len(tokens)
            self._increment_distance_tracker(state, incrementation_amount)
        elif tagging_kwargs["link"] == TagLink.BRANCH_DISTANCE:
            pass  # We don't need to do anything for branch distance, since we're not dealing with a branch presently.
        else:
            raise NotImplementedError(f"The link <{tagging_kwargs['link']}> is not currently supported.")

        return tags

    @staticmethod
    def _collect_document_unit(units: UnitCollection, structures: dict[str, list], loading_kwargs: dict[str, Any]):
        unit_id: UnitIdentifier = (loading_kwargs["collection_id"], 1)
        new_unit: TagUnit = (structures["tokens"], structures["tags"], unit_id)
        units.append(new_unit)

    @staticmethod
    def _collect_section_unit(units: UnitCollection, structures: dict[str, list], loading_kwargs: dict[str, Any],
                              section_start: int, section_end: int, partition_id: int):
        section_tokens: list[str] = structures["tokens"][section_start:section_end]
        section_tags: list[list[str]] = [stratum[section_start:section_end] for stratum in structures["tags"]]

        unit_id: UnitIdentifier = (loading_kwargs["collection_id"], partition_id)
        new_unit: TagUnit = (section_tokens, section_tags, unit_id)
        units.append(new_unit)

    @abstractmethod
    def load_data(self, input_filepath: str, loading_kwargs: dict[str, Any]) -> tuple[dict[str, list], dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def unitize_data(self, structures: dict[str, list], loading_kwargs: dict[str, Any],
                     **loaded_kwargs) -> UnitCollection:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _is_any_parallelism_incomplete(state: DocumentState) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_collection_id(filename: str, backup_id: int, loading_kwargs: dict[str, Any]) -> Union[int, str]:
        raise NotImplementedError
