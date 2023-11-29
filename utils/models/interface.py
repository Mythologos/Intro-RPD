from functools import partial
from typing import Any, Callable, Optional, Sequence, Type

from torch import mean, sum

from utils.layers.embeddings.embedding import EmbeddingLayer
from utils.layers.embeddings.chinese_bert_embedding import ChineseBertEmbedding
from utils.layers.embeddings.latin_bert_embedding import LatinBertEmbedding
from utils.layers.embeddings.latin_learned_subword_embedding import LatinLearnedSubwordEmbedding
from utils.layers.embeddings.latin_word_embedding import LatinWordEmbedding
from utils.layers.embeddings.learned_word_embedding import LearnedWordEmbedding
from utils.layers.encoders.encoder import EncoderLayer
from utils.layers.encoders.identity_encoder import IdentityEncoder
from utils.layers.encoders.lstm_encoder import LSTMEncoder
from utils.layers.encoders.transformer_encoder import TransformerEncoder
from utils.layers.modules.blender import Blender
from utils.models.constants import BlenderType, EmbeddingType, EncoderType, take_first
from utils.models.bases.encoder_crf import EncoderCRF


EMBEDDING_TABLE: dict[str, Type[EmbeddingLayer]] = {
    EmbeddingType.CHINESE_BERT: ChineseBertEmbedding,
    EmbeddingType.LATIN_BERT: LatinBertEmbedding,
    EmbeddingType.LEARNED: LearnedWordEmbedding,
    EmbeddingType.LATIN_LEARNED_SUBWORD: LatinLearnedSubwordEmbedding,
    EmbeddingType.WORD: LatinWordEmbedding
}

ENCODER_TABLE: dict[str, Type[EncoderLayer]] = {
    EncoderType.IDENTITY: IdentityEncoder,
    EncoderType.LSTM: LSTMEncoder,
    EncoderType.TRANSFORMER: TransformerEncoder
}

BLENDER_TABLE: dict[str, Optional[Callable]] = {
    BlenderType.IDENTITY: None,
    BlenderType.MEAN: partial(mean, dim=0),
    BlenderType.SUM: partial(sum, dim=0),
    BlenderType.TAKE_FIRST: take_first
}


WORD_LEVEL_EMBEDDINGS: Sequence[str] = (EmbeddingType.WORD, EmbeddingType.LEARNED)
SUBWORD_LEVEL_EMBEDDINGS: Sequence[str] = \
    (EmbeddingType.LATIN_LEARNED_SUBWORD, EmbeddingType.LATIN_BERT, EmbeddingType.CHINESE_BERT)


def build_embedding(embedding_type: str, vocabularies: dict[str, dict], **model_kwargs) -> EmbeddingLayer:
    try:
        embedding_class: Type[EmbeddingLayer] = EMBEDDING_TABLE[embedding_type]
    except KeyError:
        raise ValueError(f"The embedding type <{embedding_type}> is not supported.")

    embedding_kwargs: dict[str, Any] = {"lemmatizer": model_kwargs["lemmatizer"]}

    if embedding_type == EmbeddingType.LEARNED:
        embedding_kwargs["replacement_strategy"] = model_kwargs["replacement_strategy"]
        embedding_kwargs["replacement_probability"] = model_kwargs["replacement_probability"]

    if embedding_type in (EmbeddingType.WORD, EmbeddingType.LATIN_BERT, EmbeddingType.CHINESE_BERT):
        embedding_kwargs["pretrained_filepath"] = model_kwargs["pretrained_filepath"]
        embedding_kwargs["frozen_embeddings"] = model_kwargs["frozen_embeddings"]

    if embedding_type in (EmbeddingType.LATIN_LEARNED_SUBWORD, EmbeddingType.LATIN_BERT, EmbeddingType.CHINESE_BERT):
        embedding_kwargs["tokenizer_filepath"] = model_kwargs["tokenizer_filepath"]

    if embedding_type in (EmbeddingType.LATIN_LEARNED_SUBWORD, EmbeddingType.LEARNED, EmbeddingType.WORD):
        embedding_kwargs["input_size"] = model_kwargs["input_size"]

    embedding: EmbeddingLayer = embedding_class(vocabularies, **embedding_kwargs)
    return embedding


def build_model(components: dict[str, str], vocabularies: dict[str, dict], **model_kwargs) -> EncoderCRF:
    embedding: EmbeddingLayer = build_embedding(components["embedding"], vocabularies, **model_kwargs)
    blender: Optional[Blender] = build_blender(components["blender"])
    encoder, input_hidden_size = build_encoder(components["encoder"], embedding, **model_kwargs)
    model: EncoderCRF = EncoderCRF(components, vocabularies, embedding, blender, encoder, input_hidden_size)
    return model


def build_blender(blender_name: str) -> Optional[Blender]:
    try:
        blending_function: Callable = BLENDER_TABLE[blender_name]
    except KeyError:
        raise ValueError(f"The blender name <{blender_name}> is not recognized.")

    if blending_function is not None:
        blender: Optional[Blender] = Blender(blending_function)
    else:
        blender = None

    return blender


def build_encoder(encoder_type: str, embedding_layer: EmbeddingLayer, **model_kwargs) -> tuple[EncoderLayer, int]:
    try:
        encoder_class: Type[EncoderLayer] = ENCODER_TABLE[encoder_type]
    except KeyError:
        raise ValueError(f"The encoder type <{encoder_type}> is not supported.")

    if encoder_type == EncoderType.IDENTITY:
        encoder_kwargs: dict[str, Any] = {"input_size": 0, "hidden_size": 0, "layers": 0}
        input_hidden_size: int = embedding_layer.embedding_size
    else:
        encoder_kwargs: dict[str, Any] = {
            "input_size": embedding_layer.embedding_size,
            "hidden_size": model_kwargs["hidden_size"],
            "layers": model_kwargs["layers"]
        }

        if encoder_type == EncoderType.TRANSFORMER:
            encoder_kwargs["activation_function"] = model_kwargs["activation_function"]
            encoder_kwargs["heads"] = model_kwargs["heads"]
            input_hidden_size: int = embedding_layer.embedding_size
        else:   # encoder_type == EncoderType.LSTM:
            input_hidden_size: int = model_kwargs["hidden_size"]

    encoder: EncoderLayer = encoder_class(**encoder_kwargs)
    return encoder, input_hidden_size
