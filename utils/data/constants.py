from typing import Sequence, Union

from aenum import NamedConstant


Branch = Union[tuple[int, int], set[int]]
Parallelism = Union[set[tuple[int, int]], list[set[int]]]
ParallelismDirectory = dict[int, Parallelism]


class BranchRepresentation(NamedConstant):
    SET: str = "set"
    TUPLE: str = "tuple"


class DefinedParallelismDataset(NamedConstant):
    ASP: str = "asp"
    PSE: str = "pse-i"


DEFINED_DATASETS: Sequence[str] = tuple([dataset for dataset in DefinedParallelismDataset])   # type: ignore
