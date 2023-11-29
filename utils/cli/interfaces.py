from argparse import ArgumentParser, BooleanOptionalAction
from typing import Sequence

from utils.cli.constants import ACTIVATION_FUNCTIONS, DEFAULT_SPLITS, REPLACEMENT_STRATEGIES, VISUALIZATIONS
from utils.cli.messages import GenericMessage, NeuralMessage
from utils.data.constants import DefinedParallelismDataset
from utils.data.interface import get_dataset
from utils.data.loaders.constants import CollectionFormat, COLLECTIONS
from utils.data.tags import TAGSETS, Tagset, LINKS, TagLink
from utils.models.constants import BlenderType, BLENDERS
from utils.stats.constants import ScoringMode, SCORING_MODES
from utils.training.helper_functions import get_replacement_probability
from utils.training.optimizers import get_optimizer


def setup_parser_divisions() -> Sequence[ArgumentParser]:
    main_parser: ArgumentParser = ArgumentParser()
    subparsers = main_parser.add_subparsers(title="mode", dest="mode", required=True, help=NeuralMessage.MODE)
    train_parser: ArgumentParser = subparsers.add_parser("train")
    evaluate_parser: ArgumentParser = subparsers.add_parser("evaluate")
    return main_parser, train_parser, evaluate_parser


def add_common_optional_arguments(parser_group: ArgumentParser):
    parser_group.add_argument(
        "--collection-format", type=str, default=CollectionFormat.SECTION, choices=COLLECTIONS,
        help=GenericMessage.COLLECTION_FORMAT
    )
    parser_group.add_argument(
        "--dataset", type=get_dataset, default=DefinedParallelismDataset.ASP, help=NeuralMessage.DATASET
    )
    parser_group.add_argument(
        "--data-splits", nargs="*", type=str, default=DEFAULT_SPLITS, help=NeuralMessage.DATA_SPLITS
    )
    parser_group.add_argument(
        "--evaluation-partition", type=str, default="validation", help=NeuralMessage.EVALUATION_PARTITION
    )
    parser_group.add_argument(
        "--lemmatization", action=BooleanOptionalAction, default=False, help=NeuralMessage.LEMMATIZATION
    )
    parser_group.add_argument(
        "--link", choices=LINKS, default=TagLink.TOKEN_DISTANCE, help=GenericMessage.LINK
    )
    parser_group.add_argument("--model-location", type=str, default=None, help=GenericMessage.MODEL_LOCATION)
    parser_group.add_argument("--model-name", nargs="?", type=str, default=None, help=GenericMessage.MODEL_NAME)
    parser_group.add_argument(
        "--output-directory", nargs="?", type=str, default=None, help=NeuralMessage.OUTPUT_DIRECTORY
    )
    parser_group.add_argument(
        "--print-style", choices=["all", "checkpoint", "none"], default="checkpoint", help=NeuralMessage.PRINT_STYLE
    )
    parser_group.add_argument("--random-seed", type=int, default=42, help=GenericMessage.RANDOM_SEED)
    parser_group.add_argument(
        "--results-directory", nargs="?", type=str, default=None, help=GenericMessage.RESULTS_DIRECTORY
    )
    parser_group.add_argument("--result-display-count", type=int, default=1, help=NeuralMessage.DISPLAY_COUNT)
    parser_group.add_argument(
        "--scoring-mode", choices=SCORING_MODES, default=ScoringMode.EXACT_PARALLEL_MATCH,
        help=GenericMessage.SCORING_MODE
    )
    parser_group.add_argument("--stratum-count", type=int, default=2, help=GenericMessage.STRATUM_COUNT)
    parser_group.add_argument("--tagset", choices=TAGSETS, default=Tagset.BIO, help=GenericMessage.TAGSET)
    parser_group.add_argument("--test-filename", nargs="?", type=str, default=None, help=GenericMessage.TEST_FILENAME)
    parser_group.add_argument("--tqdm", action=BooleanOptionalAction, default=True, help=NeuralMessage.TQDM)


def add_common_training_arguments(parser_group: ArgumentParser):
    parser_group.add_argument(
        "--activation-function", "--af", type=str, choices=ACTIVATION_FUNCTIONS, default="relu",
        help=NeuralMessage.ACTIVATION_FUNCTION
    )
    parser_group.add_argument(
        "--blender", type=str, choices=BLENDERS, default=BlenderType.IDENTITY, help=NeuralMessage.BLENDER
    )
    parser_group.add_argument("--dropout", type=float, default=0.1, help=NeuralMessage.DROPOUT)
    parser_group.add_argument(
        "--embedding-filepath", nargs="?", type=str, default=None, help=NeuralMessage.EMBEDDING_FILEPATH
    )
    parser_group.add_argument("--epochs", type=int, default=None, help=NeuralMessage.EPOCHS)
    parser_group.add_argument(
        "--frozen-embeddings", action=BooleanOptionalAction, default=True, help=NeuralMessage.FROZEN_EMBEDDINGS
    )
    parser_group.add_argument("--heads", "--num-heads", nargs="?", type=int, default=1, help=NeuralMessage.HEADS)
    parser_group.add_argument("--hidden-size", type=int, default=100, help=NeuralMessage.HIDDEN_SIZE)
    parser_group.add_argument("--input-size", "--embedding-size", type=int, default=128, help=NeuralMessage.INPUT_SIZE)
    parser_group.add_argument("--layers", "--num-layers", nargs="?", type=int, default=1, help=NeuralMessage.LAYERS)
    parser_group.add_argument(
        "--lr", "--learning-rate", nargs="?", type=float, default=0.01, help=NeuralMessage.LEARNING_RATE
    )
    parser_group.add_argument(
        "--optimizer", nargs="?", type=get_optimizer, default="Adam", help=NeuralMessage.OPTIMIZER
    )
    parser_group.add_argument("--patience", type=int, default=None, help=NeuralMessage.PATIENCE)
    parser_group.add_argument("--pretrained-filepath", type=str, help=NeuralMessage.PRETRAINED_FILEPATH)
    parser_group.add_argument(
        "--replacement-probability", type=get_replacement_probability, nargs="?", default=0.5,
        help=NeuralMessage.REPLACEMENT_PROBABILITY
    )
    parser_group.add_argument(
        "--replacement-strategy", type=str, choices=REPLACEMENT_STRATEGIES, default="singleton",
        help=NeuralMessage.REPLACEMENT_STRATEGY
    )
    parser_group.add_argument("--tokenizer-filepath", type=str, help=NeuralMessage.TOKENIZER_FILEPATH)
    parser_group.add_argument(
        "--training-filename", nargs="?", type=str, default=None, help=GenericMessage.TRAINING_FILENAME
    )
    parser_group.add_argument(
        "--training-partition", type=str, default="training", help=NeuralMessage.TRAINING_PARTITION
    )
    parser_group.add_argument(
        "--validation-filename", nargs="?", type=str, default=None, help=GenericMessage.VALIDATION_FILENAME
    )
    parser_group.add_argument("--weight-decay", nargs="?", type=float, default=0, help=NeuralMessage.WEIGHT_DECAY)
    parser_group.add_argument(
        "--visualize", type=str, nargs="*", choices=VISUALIZATIONS, default=tuple(), help=NeuralMessage.VISUALIZE
    )
    parser_group.add_argument(
        "--visualization-directory", type=str, default=None, help=NeuralMessage.VISUALIZATION_DIRECTORY
    )
