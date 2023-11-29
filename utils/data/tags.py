from typing import Sequence

from aenum import Enum, NamedConstant


class BIOTag(Enum):
    INITIAL_BEGINNING: str = "B"
    INITIAL_INSIDE: str = "I"
    NONINITIAL_INSIDE: str = "J"
    OUTSIDE: str = "O"
    MIDDLE: str = "M"
    END: str = "E"
    LINKED_BEGINNING: str = "B-X"

    # The below are utility tags and are not part of the extended BIO-tagging formalism.
    START: str = "<START>"
    STOP: str = "<STOP>"


class Tagset(NamedConstant):
    BIO: str = "bio"
    BIOE: str = "bioe"
    BIOJ: str = "bioj"
    BIOM: str = "biom"
    BIOJE: str = "bioje"
    BIOMJ: str = "biomj"
    BIOME: str = "biome"
    BIOMJE: str = "biomje"


class TagLink(NamedConstant):
    BRANCH_DISTANCE: str = "bd"
    TOKEN_DISTANCE: str = "td"


TAGS: Sequence[str] = tuple([tag.value for tag in BIOTag])   # type: ignore
TAGSETS: Sequence[str] = tuple([tagset for tagset in Tagset])   # type: ignore
LINKS: Sequence[str] = tuple([link for link in TagLink])  # type: ignore

BEGINNING_TAGS: Sequence[str] = (BIOTag.INITIAL_BEGINNING.value,)
INSIDE_TAGS: Sequence[str] = (BIOTag.INITIAL_INSIDE.value, BIOTag.NONINITIAL_INSIDE.value, BIOTag.END.value)
OUTSIDE_TAGS: Sequence[str] = (BIOTag.OUTSIDE.value, BIOTag.MIDDLE.value)
FUNCTIONAL_TAGS: Sequence[str] = (BIOTag.START.value, BIOTag.STOP.value)

MIDDLE_TAGSETS: Sequence[str] = (Tagset.BIOM, Tagset.BIOME, Tagset.BIOMJ, Tagset.BIOMJE)
NONINITIAL_TAGSETS: Sequence[str] = (Tagset.BIOJ, Tagset.BIOJE, Tagset.BIOMJ, Tagset.BIOMJE)
END_TAGSETS: Sequence[str] = (Tagset.BIOE, Tagset.BIOJE, Tagset.BIOME, Tagset.BIOMJE)
