from argparse import Namespace
from json import dump
from os import mkdir, path
from random import seed
from typing import Any, Optional, TextIO, Union

from cltk.tokenizers.lat.lat import LatinWordTokenizer
from torch import cuda, load, manual_seed, save

from utils.cli.interfaces import add_common_optional_arguments, add_common_training_arguments, setup_parser_divisions
from utils.cli.messages import NeuralMessage
from utils.data.interface import compute_kneser_ney_estimation, define_data, define_vocabulary_structures, \
    get_lemmatizer, ParallelismDataset
from utils.models.constants import EMBEDDINGS, ENCODERS
from utils.models.interface import build_model
from utils.training.helper_functions import define_file_args, define_training_loop_args
from utils.training.optimizers import define_optimizer_args
from utils.visualizations.visualizers import visualize_training_outputs

from utils.training.evaluation_loop import evaluate
from utils.training.training_loop import train


if __name__ == "__main__":
    main_parser, train_parser, evaluate_parser = setup_parser_divisions()

    for subparser in (train_parser, evaluate_parser):
        subparser_required_group = subparser.add_argument_group("Required Arguments")
        subparser_required_group.add_argument("embedding", type=str, choices=EMBEDDINGS, help=NeuralMessage.EMBEDDING)
        subparser_required_group.add_argument("encoder", type=str, choices=ENCODERS, help=NeuralMessage.ENCODER)

        subparser_common_group = subparser.add_argument_group("Common Optional Arguments")
        add_common_optional_arguments(subparser_common_group)

    training_specific_group = train_parser.add_argument_group("Training-Specific Arguments")
    add_common_training_arguments(training_specific_group)

    args: Namespace = main_parser.parse_args()
    kwargs: dict[str, Any] = vars(args)

    # We provide a random seed to make computations deterministic.
    if args.print_style != "none":
        print(f"Running with random seed {args.random_seed} ...", flush=True)

    seed(args.random_seed)
    manual_seed(args.random_seed)

    # We get the appropriate device.
    device: str = "cuda" if cuda.is_available() else "cpu"

    # Preliminary steps ...
    if args.mode == "evaluate" and (args.model_location is None or not path.exists(args.model_location)):
        raise ValueError("A valid directory is required for a model in order to evaluate it. Please try again.")
    elif args.model_location is not None and path.exists(args.model_location) and not path.isdir(args.model_location):
        raise ValueError("The given filepath for results exists, but it is not a valid directory. Please try again.")

    required_partitions: list[str] = [args.evaluation_partition]
    if args.mode == "train":
        required_partitions.insert(0, args.training_partition)

    dataset_directory, dataset_loader = args.dataset

    tagging_kwargs: dict[str, Union[int, str]] = {
        "link": args.link,
        "stratum_count": args.stratum_count,
        "tagset": args.tagset
    }

    loading_kwargs: dict[str, Any] = {
        "collection_format": args.collection_format,
        "tagging_kwargs": tagging_kwargs,
        "tokenizer": LatinWordTokenizer()
    }

    current_dataset: ParallelismDataset = \
        define_data(dataset_directory, dataset_loader, args.data_splits, required_partitions, loading_kwargs)

    evaluation_kwargs: dict[str, Any] = {
        "evaluation_partition": args.evaluation_partition,
        "print_style": args.print_style,
        "result_display_count": args.result_display_count,
        "scoring_mode": args.scoring_mode,
        "tagging_kwargs": tagging_kwargs
    }

    file_args: dict[str, Union[str, TextIO, None]] = define_file_args(kwargs)

    if args.mode == "train":
        if args.print_style != "none":
            print("Setting up training...")

        # We define the optimizer's arguments.
        optimizer_args: dict[str, Any] = define_optimizer_args(kwargs["optimizer"], kwargs)

        # We define the arguments for the training loop.
        training_loop_args: dict[str, Any] = define_training_loop_args(kwargs)

        vocabulary_kwargs: dict[str, Any] = {"embedding_filepath": args.embedding_filepath}
        lemmatizer: Optional = get_lemmatizer(args.lemmatization)
        vocabularies: dict[str, dict] = define_vocabulary_structures(current_dataset, lemmatizer, **vocabulary_kwargs)
        kwargs["lemmatizer"] = lemmatizer

        if args.replacement_probability == "kneser-ney":
            kwargs["replacement_probability"] = compute_kneser_ney_estimation(vocabularies)

        components: dict[str, str] = {"blender": args.blender, "embedding": args.embedding, "encoder": args.encoder}

        # We instantiate the model and define the location at which it will be saved.
        model = build_model(components, vocabularies, **kwargs)
        model.to(device)

        # We train the model.
        trained_model, training_outputs = train(
            model, device, current_dataset, kwargs["optimizer"], optimizer_args, training_loop_args,
            evaluation_kwargs, file_args
        )

        best_results_display: str = training_outputs["best_scoring_structure"].get_statistics_display()
        best_results_output: str = f"Overall Training Results - Best Model Statistics:\n{best_results_display}\n"

        if args.print_style != "none":
            print(best_results_output)

        if file_args["validation_file"] is not None:
            file_args["validation_file"].write(best_results_output)

        # We save the model deemed best by the training process.
        if file_args["model_location"] is not None:
            save(trained_model, file_args["model_location"])
            model_output_file: TextIO = open(file_args["model_outputs_location"], encoding="utf-8", mode="w+")
            del training_outputs["best_scoring_structure"]
            dump(training_outputs, model_output_file, indent=1)
            model_output_file.close()

        # Finally, if there are any visualizations specified, we create and save them.
        if len(args.visualize) > 0 and args.visualization_directory is not None:
            visualization_directory_path: str = args.visualization_directory
            if args.model_name is not None:
                visualization_directory_path += f"/{args.model_name}"

            if not path.exists(visualization_directory_path):
                mkdir(visualization_directory_path, mode=711)
            elif path.exists(visualization_directory_path) and not path.isdir(visualization_directory_path):
                raise ValueError("Invalid path for saving visuals. Please try again.")
            visualize_training_outputs(training_outputs, args.visualize, visualization_directory_path)
    elif args.mode == "evaluate":
        if args.print_style != "none":
            print("Starting evaluation...")

        model = load(file_args["model_location"], map_location=device)
        evaluation_outputs: dict[str, Any] = evaluate(
            model, device, current_dataset, evaluation_kwargs, file_args, "test_file", args.tqdm
        )

        overall_results_display: str = f"Overall Test Results:" \
                                       f"\n\t* Precision: {evaluation_outputs['precision']}" \
                                       f"\n\t* Recall: {evaluation_outputs['recall']}" \
                                       f"\n\t* F1: {evaluation_outputs['f1']}"

        if args.print_style != "none":
            print(overall_results_display)
    else:
        raise ValueError(f"The mode <{args.mode}> is not supported.")

    for (key, value) in file_args.items():
        if isinstance(value, TextIO):
            value.close()
