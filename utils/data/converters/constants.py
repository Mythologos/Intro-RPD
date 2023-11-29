from dataclasses import dataclass
from typing import Union

from aenum import Enum


FlagDict = dict[str, Union[bool, str]]

SERMON_TITLE_REGEX: str = r"<(?P<primary_id>[\d\w]+(?:[\s]+augm.)?)[\s]*(?P<secondary_id>[\d\w]*)>"


class XMLSpecialCharacter(Enum, init='default xml'):
    AMPERSAND = ("&", "&amp;")
    GREATER_THAN = (">", "&gt;")
    LESS_THAN = ("<", "&lt;")
    DOUBLE_QUOTE = ("\"", "&quot;")
    APOSTROPHE = ("\'", "&apos;")


@dataclass(order=True)
class BratAnnotation:
    start: int
    end: int
    parallelism_id: int
