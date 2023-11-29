from typing import Any, Callable, Sequence, Union

from aenum import NamedConstant
from numpy import arange, iinfo, int32, linspace

from utils.data.interface import DefinedParallelismDataset
from utils.data.tags import TagLink, Tagset
from utils.models.constants import BlenderType, EmbeddingType, EncoderType
from utils.stats.constants import ScoringMode


HyperparameterConstraintSpace = dict[tuple[str, str], list[tuple[str, Callable]]]
HyperparameterDefaultSpace = dict[str, Any]
HyperparameterSpace = dict[str, list[Any]]

ACTIVATION_RANGE: Sequence[str] = ("relu", "gelu")
BIDIRECTIONAL_RANGE: Sequence[str] = ("--bidirectional", "--no-bidirectional")
BLENDER_RANGE: Sequence[str] = tuple(
    blender for blender in BlenderType if blender != BlenderType.IDENTITY   # type: ignore
)
DROPOUT_RANGE: Sequence[float] = linspace(0.0, 0.5, num=11)
EPOCHS_RANGE: Sequence[int] = [i for i in range(50, 500, 25)]
FROZEN_EMBEDDING_RANGE: Sequence[str] = ("--frozen-embeddings", "--no-frozen-embeddings")
HEAD_RANGE: Sequence[int] = (1, 2, 4, 8)
HIGHER_WORD_EMBEDDING_DIM_RANGE: Sequence[int] = (128, 192, 256, 384, 512, 768, 1024)
HIGHER_WORD_HIDDEN_DIM_RANGE: Sequence[int] = (256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048)
LAYER_RANGE: Sequence[int] = arange(1, 5, 1)
LEARNING_RATE_RANGE: Sequence[float] = linspace(0.0001, 0.01, num=100)
LEMMATIZATION_RANGE: Sequence[str] = ("--lemmatization", "--no-lemmatization")
LINK_RANGE: Sequence[str] = tuple(TagLink)   # type: ignore
LOWER_WORD_EMBEDDING_DIM_RANGE: Sequence[int] = (64, 96, 128, 192, 256, 384, 512, 768)
LOWER_WORD_HIDDEN_DIM_RANGE: Sequence[int] = (32, 48, 64, 96, 128, 192, 256, 384, 512)
MATCH_DIRECTION_RANGE: Sequence[str] = ("inward", "outward")
# MATCHER_RANGE: Sequence[str] = tuple(key for key in MATCHERS.keys())
MAXIMUM_SPAN_DISTANCE_RANGE: Sequence[int] = arange(1, 21, 1)
MAXIMUM_SPAN_SIZE_RANGE: Sequence[int] = arange(5, 26, 1)
OPTIMIZER_RANGE: Sequence[str] = \
    ["ASGD", "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"]
PATIENCE_RANGE: Sequence[int] = [i for i in range(0, 500, 25)]
RANDOM_SEED_RANGE: Sequence[int] = linspace(0, iinfo(int32).max)
REPLACEMENT_PROBABILITY_RANGE: Sequence[Union[float, str]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "kneser-ney")
REPLACEMENT_STRATEGY_NAMES: Sequence[str] = ("any", "singleton", "none")
SAMPLING_MULTIPLIER_RANGE: Sequence[int] = (1, 5, 10, 15, 20, 25)
SCORING_MODE_RANGE: Sequence[str] = list(ScoringMode)   # type: ignore
STRATUM_RANGE: Sequence[int] = (1, 2)
TAGSET_RANGE: list[str] = tuple(Tagset)   # type: ignore
WEIGHT_DECAY_RANGE: Sequence[float] = linspace(0.0, 0.1, num=1000)
WORD_EMBEDDING_RANGE: Sequence[str] = ("latin_w2v_bamman_lemma300_100_1", "latin_w2v_bamman_lemma_tt")


class Hyperparameter(NamedConstant):
    ACTIVATION: str = "activation"
    BIDIRECTIONAL: str = "bidirectional"
    BLENDER: str = "blender"
    COLLECTION_FORMAT: str = "collection-format"
    DATASET: str = "dataset"
    DROPOUT: str = "dropout"
    EPOCHS: str = "epochs"
    FROZEN_EMBEDDINGS: str = "frozen-embeddings"
    HEADS: str = "heads"
    LAYERS: str = "layers"
    LEARNING_RATE: str = "learning-rate"
    LEMMATIZATION: str = "lemmatization"
    LINK: str = "link"
    OPTIMIZER: str = "optimizer"
    PATIENCE: str = "patience"
    RANDOM_SEED: str = "random-seed"
    REPLACEMENT_PROBABILITY: str = "replacement-probability"
    REPLACEMENT_STRATEGY: str = "replacement-strategy"
    SCORING_MODE: str = "scoring-mode"
    STRATUM_COUNT: str = "stratum-count"
    TAGSET: str = "tagset"
    WEIGHT_DECAY: str = "weight-decay"
    WORD_EMBEDDINGS: str = "embeddings"
    WORD_EMBEDDING_DIMENSIONALITY: str = "embedding-dim-first"
    WORD_HIDDEN_DIMENSIONALITY: str = "hidden-dim-first"


HYPERPARAMETERS: Sequence[str] = [current_hyperparameter for current_hyperparameter in Hyperparameter]   # type: ignore


def check_word_embedding_dim(chosen_hyperparameters: dict[str, Any]) -> bool:
    constraint_satisfied: bool = True
    if chosen_hyperparameters[Hyperparameter.WORD_EMBEDDINGS] == "latin_w2v_bamman_lemma300_100_1":
        chosen_hyperparameters[Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY] = 300
    elif chosen_hyperparameters[Hyperparameter.WORD_EMBEDDINGS] == "latin_w2v_bamman_lemma_tt":
        chosen_hyperparameters[Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY] = 50
    else:
        constraint_satisfied = False
    return constraint_satisfied


def check_word_state_compression(chosen_hyperparameters: dict[str, Any]) -> bool:
    constraint_satisfied: bool = False
    if chosen_hyperparameters[Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY] >= \
            chosen_hyperparameters[Hyperparameter.WORD_HIDDEN_DIMENSIONALITY]:
        constraint_satisfied = True
    return constraint_satisfied


HYPERPARAMETER_SPACES: dict[Union[str, tuple[str, str]], HyperparameterSpace] = {
    "globals": {
        Hyperparameter.COLLECTION_FORMAT: ["section", "document"],
        Hyperparameter.EPOCHS: EPOCHS_RANGE,
        Hyperparameter.LEARNING_RATE: LEARNING_RATE_RANGE,
        Hyperparameter.LINK: LINK_RANGE,
        Hyperparameter.OPTIMIZER: OPTIMIZER_RANGE,
        Hyperparameter.PATIENCE: PATIENCE_RANGE,
        Hyperparameter.RANDOM_SEED: RANDOM_SEED_RANGE,
        Hyperparameter.SCORING_MODE: SCORING_MODE_RANGE,
        Hyperparameter.STRATUM_COUNT: STRATUM_RANGE,
        Hyperparameter.TAGSET: TAGSET_RANGE,
        Hyperparameter.WEIGHT_DECAY: WEIGHT_DECAY_RANGE
    },
    (EmbeddingType.LEARNED, EncoderType.LSTM): {
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
        Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY: LOWER_WORD_EMBEDDING_DIM_RANGE,
        Hyperparameter.REPLACEMENT_PROBABILITY: REPLACEMENT_PROBABILITY_RANGE,
        Hyperparameter.REPLACEMENT_STRATEGY: REPLACEMENT_STRATEGY_NAMES,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: LOWER_WORD_HIDDEN_DIM_RANGE
    },
    (EmbeddingType.WORD, EncoderType.LSTM): {
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.WORD_EMBEDDINGS: WORD_EMBEDDING_RANGE,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: LOWER_WORD_HIDDEN_DIM_RANGE
    },
    (EmbeddingType.LATIN_LEARNED_SUBWORD, EncoderType.TRANSFORMER): {
        Hyperparameter.ACTIVATION: ACTIVATION_RANGE,
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.DROPOUT: DROPOUT_RANGE,
        Hyperparameter.HEADS: HEAD_RANGE,
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
        Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY: HIGHER_WORD_EMBEDDING_DIM_RANGE,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: HIGHER_WORD_HIDDEN_DIM_RANGE
    },
    (EmbeddingType.LATIN_BERT, EncoderType.IDENTITY): {
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
    },
    (EmbeddingType.LATIN_BERT, EncoderType.LSTM): {
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: LOWER_WORD_HIDDEN_DIM_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
    },
    (EmbeddingType.LATIN_BERT, EncoderType.TRANSFORMER): {
        Hyperparameter.ACTIVATION: ACTIVATION_RANGE,
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.DROPOUT: DROPOUT_RANGE,
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.HEADS: HEAD_RANGE,
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: HIGHER_WORD_HIDDEN_DIM_RANGE
    },
    (EmbeddingType.CHINESE_BERT, EncoderType.IDENTITY): {
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
    },
    (EmbeddingType.CHINESE_BERT, EncoderType.LSTM): {
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: LOWER_WORD_HIDDEN_DIM_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
    },
    (EmbeddingType.CHINESE_BERT, EncoderType.TRANSFORMER): {
        Hyperparameter.ACTIVATION: ACTIVATION_RANGE,
        Hyperparameter.BLENDER: BLENDER_RANGE,
        Hyperparameter.DROPOUT: DROPOUT_RANGE,
        Hyperparameter.FROZEN_EMBEDDINGS: FROZEN_EMBEDDING_RANGE,
        Hyperparameter.HEADS: HEAD_RANGE,
        Hyperparameter.LAYERS: LAYER_RANGE,
        Hyperparameter.LEMMATIZATION: LEMMATIZATION_RANGE,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: HIGHER_WORD_HIDDEN_DIM_RANGE
    },
}


HYPERPARAMETER_DEFAULTS: dict[Union[str, tuple[str, str]], HyperparameterDefaultSpace] = {
    "globals": {
        Hyperparameter.COLLECTION_FORMAT: "section",
        Hyperparameter.DATASET: DefinedParallelismDataset.ASP,
        Hyperparameter.EPOCHS: 200,
        Hyperparameter.LEARNING_RATE: .01,
        Hyperparameter.LINK: TagLink.TOKEN_DISTANCE,
        Hyperparameter.OPTIMIZER: "Adam",
        Hyperparameter.PATIENCE: 25,
        Hyperparameter.RANDOM_SEED: 42,
        Hyperparameter.SCORING_MODE: ScoringMode.MAX_BRANCH_AWARE_WORD_OVERLAP,
        Hyperparameter.STRATUM_COUNT: 2,
        Hyperparameter.TAGSET: Tagset.BIO,
        Hyperparameter.WEIGHT_DECAY: 0.0
    },
    (EmbeddingType.LEARNED, EncoderType.LSTM): {
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.REPLACEMENT_PROBABILITY: "kneser-ney",
        Hyperparameter.REPLACEMENT_STRATEGY: "singleton",
        Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY: 128,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: 96
    },
    (EmbeddingType.WORD, EncoderType.LSTM): {
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--lemmatization",
    },
    (EmbeddingType.LATIN_LEARNED_SUBWORD, EncoderType.TRANSFORMER): {
        Hyperparameter.ACTIVATION: "relu",
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.DROPOUT: 0.1,
        Hyperparameter.HEADS: 1,
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.WORD_EMBEDDING_DIMENSIONALITY: 256,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: 512
    },
    (EmbeddingType.LATIN_BERT, EncoderType.IDENTITY): {
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.LEMMATIZATION: "--no-lemmatization"
    },
    (EmbeddingType.LATIN_BERT, EncoderType.LSTM): {
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: 256
    },
    (EmbeddingType.LATIN_BERT, EncoderType.TRANSFORMER): {
        Hyperparameter.ACTIVATION: "relu",
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.DROPOUT: 0.1,
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.HEADS: 1,
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: 1024
    },
    (EmbeddingType.CHINESE_BERT, EncoderType.IDENTITY): {
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.DATASET: DefinedParallelismDataset.PSE,
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.STRATUM_COUNT: 1
    },
    (EmbeddingType.CHINESE_BERT, EncoderType.LSTM): {
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.DATASET: DefinedParallelismDataset.PSE,
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.STRATUM_COUNT: 1,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: 256
    },
    (EmbeddingType.CHINESE_BERT, EncoderType.TRANSFORMER): {
        Hyperparameter.ACTIVATION: "relu",
        Hyperparameter.BLENDER: BlenderType.MEAN,
        Hyperparameter.DATASET: DefinedParallelismDataset.PSE,
        Hyperparameter.DROPOUT: 0.1,
        Hyperparameter.FROZEN_EMBEDDINGS: "--frozen-embeddings",
        Hyperparameter.HEADS: 1,
        Hyperparameter.LAYERS: 1,
        Hyperparameter.LEMMATIZATION: "--no-lemmatization",
        Hyperparameter.STRATUM_COUNT: 1,
        Hyperparameter.WORD_HIDDEN_DIMENSIONALITY: 1024
    }
}

HYPERPARAMETER_CONSTRAINTS: HyperparameterConstraintSpace = {
    (EmbeddingType.LEARNED, EncoderType.LSTM): [
        ("word_dim_compression", check_word_state_compression)
    ],
    (EmbeddingType.WORD, EncoderType.LSTM): [
        ("word_embedding", check_word_embedding_dim),
        ("word_dim_compression", check_word_state_compression)
    ],
    (EmbeddingType.LATIN_LEARNED_SUBWORD, EncoderType.TRANSFORMER): [],
    (EmbeddingType.LATIN_BERT, EncoderType.IDENTITY): [],
    (EmbeddingType.LATIN_BERT, EncoderType.LSTM): [],
    (EmbeddingType.LATIN_BERT, EncoderType.TRANSFORMER): [],
    (EmbeddingType.CHINESE_BERT, EncoderType.IDENTITY): [],
    (EmbeddingType.CHINESE_BERT, EncoderType.LSTM): [],
    (EmbeddingType.CHINESE_BERT, EncoderType.TRANSFORMER): []
}

HYPERPARAMETER_TEXT_MAPPINGS: dict[str, dict[str, Any]] = {
    Hyperparameter.BIDIRECTIONAL: {"True": "--bidirectional", "False": "--no-bidirectional"},
    Hyperparameter.FROZEN_EMBEDDINGS: {"True": "--frozen-embeddings", "False": "--no-frozen-embeddings"},
    Hyperparameter.LEMMATIZATION: {"True": "--lemmatization", "False": "--no-lemmatization"}
}


CONSTRAINT_TABLE: dict[str, Callable] = {
    "word-embedding": check_word_embedding_dim,
    "word-dim-compression": check_word_state_compression
}

CONSTRAINT_KEYS: Sequence[str] = tuple([key for key in CONSTRAINT_TABLE.keys()])
