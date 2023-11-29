from typing import Sequence

from aenum import NamedConstant

from torch import Tensor


class EmbeddingType(NamedConstant):
    CHINESE_BERT: str = "chinese-bert"
    LATIN_BERT: str = "latin-bert"
    LEARNED: str = "learned"
    LATIN_LEARNED_SUBWORD: str = "latin-learned-subword"
    WORD: str = "word"


class EncoderType(NamedConstant):
    IDENTITY: str = "identity"
    LSTM: str = "lstm"
    TRANSFORMER: str = "transformer"


class BlenderType(NamedConstant):
    IDENTITY: str = "identity"
    MEAN: str = "mean"
    SUM: str = "sum"
    TAKE_FIRST: str = "take-first"


BLENDERS: Sequence[str] = tuple([blender for blender in BlenderType])   # type: ignore
EMBEDDINGS: Sequence[str] = tuple([embedding for embedding in EmbeddingType])   # type: ignore
ENCODERS: Sequence[str] = tuple([encoder for encoder in EncoderType])   # type: ignore


def take_first(input_tensor: Tensor) -> Tensor:
    return input_tensor[0]
