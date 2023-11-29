from aenum import Enum


UNK_TOKEN: str = "<unknown>"   # formatted to be in line with Gensim's Word2Vec UNK token.


class BERTSpecialToken(Enum, init='value token'):
    PADDING_TOKEN: str = (0, "[PAD]")
    UNKNOWN_TOKEN: str = (1, "[UNK]")
    CLASS_TOKEN: str = (2, "[CLS]")
    SEPARATION_TOKEN: str = (3, "[SEP]")
    MASKING_TOKEN: str = (4, "[MASK]")


STARTING_VOCABULARY: dict[str, int] = {
    BERTSpecialToken.PADDING_TOKEN.token: BERTSpecialToken.PADDING_TOKEN.value,
    BERTSpecialToken.UNKNOWN_TOKEN.token: BERTSpecialToken.UNKNOWN_TOKEN.value,
    BERTSpecialToken.CLASS_TOKEN.token: BERTSpecialToken.CLASS_TOKEN.value,
    BERTSpecialToken.SEPARATION_TOKEN.token: BERTSpecialToken.SEPARATION_TOKEN.value,
    BERTSpecialToken.MASKING_TOKEN.token: BERTSpecialToken.MASKING_TOKEN.value
}
