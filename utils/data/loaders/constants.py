from dataclasses import dataclass
from typing import Any, Sequence, Union, Optional

from aenum import NamedConstant
from cltk.tokenizers import LatinWordTokenizer

DocumentState = dict[str, Any]
UnitIdentifier = tuple[Union[str, int], int]
TagUnit = tuple[list[str], list[list[str]], UnitIdentifier]
UnitCollection = list[TagUnit]
ParallelismDataset = dict[str, UnitCollection]

PSEFileGroup = tuple[str, Optional[tuple[str, str]], str]


class CollectionFormat(NamedConstant):
    DOCUMENT: str = "document"
    SECTION: str = "section"


COLLECTIONS: Sequence[str] = tuple([collection_format for collection_format in CollectionFormat])   # type: ignore


@dataclass
class PSEWord:
    word: str
    paragraph_index: int
    sentence_index: int
    word_index: int
    part_of_speech: str
    named_entity: str
    dependency_parent: str
    dependency_tag: str
    spaces: int = 0
    parallelism_id: Optional[int] = None
    branch_id: Optional[int] = None
    alternate_parallelism_id: Optional[int] = None
    alternate_branch_id: Optional[int] = None


CUSTOM_LATIN_EXCEPTIONS: list[str] = [
    "afflictione", "assumptione", "carne", "compunctione", "confessione", "confusione",
    "consuetudine", "conuersatione", "conuersione", "dilectione"
]
EXCEPTIONS: list[str] = LatinWordTokenizer.EXCEPTIONS + CUSTOM_LATIN_EXCEPTIONS

NORMALIZATION_TABLE: dict[str, str] = {
    u'，': u',',
    u'。': u'.',
    u'！': u'.',
    u'!': u'.',
    u'？': u'.',
    u'?': u'.',
    u';': u'.',
    u'；': u'.',
    u'“': u'\"',
    u'”': u'\"'
}

PSE_PARAGRAPH_TEMPLATE: str = "<Paragraph {0}>"
PSE_DESIGNATED_LINE_BEGINNING: str = ">>> "
PSE_ENDING_PUNCTUATION: Sequence[str] = (u'，', u',', u'。', u'.', u'！', u'.', u'!', u'？', u'?', u';', u'；')
