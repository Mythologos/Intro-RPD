from math import inf
from os import path
from typing import Any, TextIO, Union


def define_training_loop_args(general_kwargs: dict[str, Any]) -> dict[str, Any]:
    training_loop_kwargs: dict[str, Any] = {}

    # We can define a maximum number of epochs, a patience, or both.
    # If epochs is defined (but patience is not),
    # then there's just a maximum number of epochs for which training should occur.
    # If patience is defined (but epochs is not),
    # then there's no upper limit on the maximum number of epochs, but there is an eventual convergence.
    # If both are defined, then this is effectively an early stopping strategy.
    if general_kwargs.get("epochs", None) is None and general_kwargs.get("patience", None) is None:
        raise ValueError("At least one of --epochs or --patience must be defined. Please try again.")
    else:
        if general_kwargs.get("epochs", None) is not None:
            training_loop_kwargs["epochs"] = general_kwargs["epochs"]
        else:
            training_loop_kwargs["epochs"] = inf

        if general_kwargs.get("patience", None) is not None:
            training_loop_kwargs["patience"] = general_kwargs["patience"]
        else:
            training_loop_kwargs["patience"] = inf

    if general_kwargs.get("loss_functions", None) is not None:
        training_loop_kwargs["loss_functions"] = general_kwargs["loss_functions"]

    training_loop_kwargs["tqdm"] = general_kwargs["tqdm"]
    training_loop_kwargs["print_style"] = general_kwargs["print_style"]
    training_loop_kwargs["training_partition"] = general_kwargs["training_partition"]
    return training_loop_kwargs


def define_file_args(provided_kwargs: dict[str, Any]) -> dict[str, Union[str, TextIO, None]]:
    current_file_kwargs: dict[str, Union[str, TextIO, None]] = {
        "training_file": None,
        "validation_file": None,
        "test_file": None,
        "model_location": None,
        "model_output_location": None,
        "output_directory": None
    }

    if provided_kwargs.get("results_directory", None) is not None:
        output_filenames: list[tuple[str, str]] = [
            (provided_kwargs.get("training_filename", None), "training_file"),
            (provided_kwargs.get("validation_filename", None), "validation_file"),
            (provided_kwargs.get("test_filename", None), "test_file")
        ]

        for (filename, file_type) in output_filenames:
            if filename is not None:
                new_filepath: str = f"{provided_kwargs['results_directory']}/{filename}"
                new_filepath += ".txt"
                current_file_kwargs[file_type] = open(new_filepath, encoding="utf-8", mode="w+")

    if provided_kwargs["mode"] == "evaluate" and \
            (not provided_kwargs["model_location"] or not provided_kwargs["model_name"]):
        raise ValueError(f"A filename must be supplied for the <{provided_kwargs['mode']}> mode. Please try again.")
    elif provided_kwargs["model_location"] and provided_kwargs["model_name"]:
        model_filepath: str = f"{provided_kwargs['model_location']}/{provided_kwargs['model_name']}.model"
        model_outputs_filepath: str = f"{provided_kwargs['model_location']}/{provided_kwargs['model_name']}.json"
        current_file_kwargs["model_location"] = model_filepath
        current_file_kwargs["model_outputs_location"] = model_outputs_filepath

    if provided_kwargs.get("output_directory", None) is not None:
        if path.isdir(provided_kwargs["output_directory"]):
            current_file_kwargs["output_directory"] = provided_kwargs["output_directory"]
        else:
            raise ValueError("The path provided for output values is not a directory.")

    return current_file_kwargs


def get_replacement_probability(replacement_value: str):
    if not (isinstance(replacement_value, float) or isinstance(replacement_value, str)):
        raise TypeError(f"The value {replacement_value} is not supported.")
    elif replacement_value.startswith(".") and replacement_value[1:].isdecimal():
        replacement_value = float(replacement_value)
    elif isinstance(replacement_value, str) and replacement_value != "kneser-ney":
        raise TypeError(f"Unrecognized string option.")
    return replacement_value
