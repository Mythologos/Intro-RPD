from argparse import Action, ArgumentParser, Namespace
from copy import deepcopy
from random import choice, seed, randint
from sys import maxsize
from typing import Any, Callable, Optional, Sequence, TextIO, Union

from natsort import natsorted

from utils.cli.messages import HyperparameterMessage, GenericMessage
from utils.models.constants import EMBEDDINGS, ENCODERS
from utils.optimization.hyperparameters import CONSTRAINT_TABLE, HYPERPARAMETERS, HYPERPARAMETER_CONSTRAINTS, \
    HYPERPARAMETER_DEFAULTS, HYPERPARAMETER_SPACES, HYPERPARAMETER_TEXT_MAPPINGS, CONSTRAINT_KEYS
from utils.optimization.formats import BASH_FORMAT, EVALUATION_OUTPUT_FORMATS, TRAINING_OUTPUT_FORMATS, \
    get_numeric_string


class HyperparameterParseAction(Action):
    def __init__(self, option_strings, dest, nargs, **kwargs):
        super().__init__(option_strings, dest, nargs, **kwargs)

    def __call__(self, hyperparameter_parser: ArgumentParser, namespace: Namespace,
                 values: Union[str, Sequence, None], option_string=None, *hyper_args, **hyper_kwargs):
        hyperparameters: Sequence[str] = HYPERPARAMETERS
        specified_hyperparameters: dict[str, Any] = {}

        if len(values) % 2 != 0:
            raise ValueError("The number of values in the list is not even. "
                             "Currently, this method only supports one-to-one hyperparameter-argument pairs.")

        value_index: int = 0
        while value_index < len(values):
            if values[value_index] in hyperparameters:
                hyperparameter_name: str = values[value_index]
                if value_index + 1 < len(values) and values[value_index + 1] not in hyperparameters:
                    if HYPERPARAMETER_TEXT_MAPPINGS.get(hyperparameter_name) is not None:
                        mapped_value: Any = HYPERPARAMETER_TEXT_MAPPINGS[hyperparameter_name] \
                            .get(values[value_index + 1], None)
                        if mapped_value is not None:
                            specified_value = mapped_value
                        else:
                            raise ValueError(f"The given mapping value, {mapped_value}, "
                                             f"is not valid for {hyperparameter_name}.")
                    else:
                        specified_value = values[value_index + 1]
                    specified_hyperparameters[hyperparameter_name] = specified_value
                elif value_index + 1 > len(values):
                    raise ValueError(f"The last hyperparameter, <{hyperparameter_name}>, "
                                     f"does not have a corresponding value.")
                else:  # value_index + 1 < len(values) and values[value_index + 1] in hyperparameters:
                    raise ValueError(f"The hyperparameter <{hyperparameter_name}> is missing a corresponding value. "
                                     f"Please try again.")
            else:
                raise ValueError(f"An invalid series of arguments was given; "
                                 f"<{values[value_index]}> is not recognized in that position. Please try again.")
            value_index += 2
        setattr(namespace, self.dest, specified_hyperparameters)


def reorganize_hyperparameters(unorganized_hyperparameters: dict[str, Any]) -> list[Any]:
    hyperparameters: list[Any] = list(unorganized_hyperparameters.items())
    sorted_hyperparameters = natsorted(hyperparameters, key=lambda x: x[0])
    organized_hyperparameters = [value for (key, value) in sorted_hyperparameters]
    return organized_hyperparameters


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("embedding", type=str, choices=EMBEDDINGS, help=HyperparameterMessage.EMBEDDING)
    parser.add_argument("encoder", type=str, choices=ENCODERS, help=HyperparameterMessage.ENCODER)
    parser.add_argument(
        "--varied-hyperparameters", type=str, nargs="*", choices=HYPERPARAMETERS,
        help=HyperparameterMessage.VARIED
    )
    parser.add_argument(
        "--specified-hyperparameters", action=HyperparameterParseAction, nargs="*",
        default={}, help=HyperparameterMessage.SPECIFIED
    )
    parser.add_argument(
        "--added-constraints", type=str, nargs="*", choices=CONSTRAINT_KEYS,
        default=tuple(), help=HyperparameterMessage.CONSTRAINTS
    )
    parser.add_argument("--model-location", type=str, default="models", help=GenericMessage.MODEL_LOCATION)
    parser.add_argument("--model-name", type=str, default="model", help=GenericMessage.MODEL_NAME)
    parser.add_argument("--results-directory", type=str, default="results", help=GenericMessage.RESULTS_DIRECTORY)
    parser.add_argument("--training-filename", type=str, default="training", help=GenericMessage.TRAINING_FILENAME)
    parser.add_argument(
        "--validation-filename", type=str, default="validation", help=GenericMessage.VALIDATION_FILENAME
    )
    parser.add_argument("--test-filename", type=str, default="optimization", help=GenericMessage.TEST_FILENAME)
    parser.add_argument("--test-partition", type=str, default="optimization", help=HyperparameterMessage.TEST_PARTITION)
    parser.add_argument("--seed", type=int, default=randint(0, maxsize), help=GenericMessage.RANDOM_SEED)
    parser.add_argument("--trials", type=int, default=32, help=HyperparameterMessage.TRIALS)
    parser.add_argument("--trial-start-offset", type=int, default=0, help=HyperparameterMessage.TRIAL_OFFSET)
    parser.add_argument("--output-filename", type=str, help=HyperparameterMessage.OUTPUT_FILENAME)
    parser.add_argument(
        "--output-format", type=str, choices=["text", "bash"], default="text", help=HyperparameterMessage.OUTPUT_FORMAT
    )
    args: Namespace = parser.parse_args()
    seed(args.seed)

    components: tuple[str, str] = (args.embedding, args.encoder)
    try:
        selected_training_format: str = TRAINING_OUTPUT_FORMATS[components]
        selected_evaluation_format: str = EVALUATION_OUTPUT_FORMATS[components]
        model_specific_hyperparameters: set[str] = set(HYPERPARAMETER_SPACES[components].keys())
        model_specific_defaults: set[str] = set(HYPERPARAMETER_DEFAULTS[components].keys())
    except KeyError:
        raise ValueError(f"The model with <{components}> is not yet fully supported.")
    except BaseException:
        raise NotImplementedError("The raised exception was not anticipated.")

    full_hyperparameter_set_spaces: set[str] = set(HYPERPARAMETER_SPACES["globals"].keys())
    full_hyperparameter_set_defaults: set[str] = set(HYPERPARAMETER_DEFAULTS["globals"].keys())
    full_hyperparameter_set: set[str] = full_hyperparameter_set_spaces.union(full_hyperparameter_set_defaults)
    full_hyperparameter_set = full_hyperparameter_set.union(model_specific_hyperparameters)
    full_hyperparameter_set = full_hyperparameter_set.union(model_specific_defaults)

    current_trial: int = 1
    trial_hyperparameters: list[dict[str, Any]] = []
    trial_commands: list[str] = []
    trial_eval_commands: list[str] = []
    while current_trial <= args.trials:
        numeric_string_segment: str = f"{get_numeric_string(current_trial + args.trial_start_offset, 3)}"
        model_name: str = f"{args.model_name}_{numeric_string_segment}"
        training_filename: str = f"{args.training_filename}_{numeric_string_segment}"
        validation_filename: str = f"{args.validation_filename}_{numeric_string_segment}"
        test_filename: str = f"{args.test_filename}_{numeric_string_segment}"
        training_file_hyperparameters: list[str] = \
            [args.model_location, model_name, args.results_directory, training_filename, validation_filename]
        test_file_hyperparameters: list[str] = \
            [args.model_location, model_name, args.results_directory, test_filename, args.test_partition]

        chosen_hyperparameters: dict[str, Any] = {}
        for hyperparameter in full_hyperparameter_set:
            if hyperparameter in args.varied_hyperparameters:
                # We get a value from the model's hyperparameter space.
                model_hyperparameter_space: Optional[Sequence[Any]] = \
                    HYPERPARAMETER_SPACES[components].get(hyperparameter, None)
                if model_hyperparameter_space is None:
                    model_hyperparameter_space: Optional[Sequence[Any]] = \
                        HYPERPARAMETER_SPACES["globals"].get(hyperparameter, None)
                    if model_hyperparameter_space is None:
                        raise ValueError(f"The value {hyperparameter} is not defined for <{components}>'s "
                                         f"hyperparameter space.")

                chosen_hyperparameters[hyperparameter] = choice(model_hyperparameter_space)
            elif hyperparameter in args.specified_hyperparameters:
                chosen_hyperparameters[hyperparameter] = args.specified_hyperparameters[hyperparameter]
            else:
                # We get the model's default.
                model_default: Any = HYPERPARAMETER_DEFAULTS[components].get(hyperparameter, None)
                if model_default is None:
                    model_default = HYPERPARAMETER_DEFAULTS["globals"].get(hyperparameter, None)
                    if model_default is None:
                        raise ValueError(f"The value {hyperparameter} is not defined for <{components}>.")

                chosen_hyperparameters[hyperparameter] = model_default

        constraints: list[tuple[str, Callable]] = deepcopy(HYPERPARAMETER_CONSTRAINTS[components])

        for constraint_name in args.added_constraints:
            constraints.append((constraint_name.replace("-", "_"), CONSTRAINT_TABLE[constraint_name]))

        for (constraint, constraint_function) in constraints:
            constraint_satisfied: bool = constraint_function(chosen_hyperparameters)
            if constraint_satisfied is False:
                break
        else:
            if chosen_hyperparameters not in trial_hyperparameters:
                trial_hyperparameters.append(chosen_hyperparameters)
                chosen_copy: dict[str, Any] = deepcopy(chosen_hyperparameters)
                reorganized_hyperparameters: list[Any] = reorganize_hyperparameters(chosen_copy)
                training_command: str = \
                    selected_training_format.format(*training_file_hyperparameters, *reorganized_hyperparameters)
                evaluation_command: str = \
                    selected_evaluation_format.format(*test_file_hyperparameters, *reorganized_hyperparameters)

                if args.output_filename is not None:
                    trial_commands.append(f"{training_command}\n")
                    trial_eval_commands.append(f"{evaluation_command}\n")
                else:
                    print(training_command)
                    print(evaluation_command)

                current_trial += 1

    if args.output_filename is not None:
        if args.output_format == "text":
            output_filename: str = f"{args.output_filename}.txt"
            output_file: Optional[TextIO] = open(output_filename, encoding="utf-8", mode="w+")
            for command in trial_commands:
                output_file.write(command)
            output_file.close()
        elif args.output_format == "bash":
            current_bash_trial: int = 1
            while current_bash_trial <= args.trials:
                numeric_string_segment: str = get_numeric_string(current_bash_trial + args.trial_start_offset, 3)
                bash_filename: str = f"{args.output_filename}_{numeric_string_segment}.sh"
                bash_file: TextIO = open(bash_filename, encoding="utf-8", mode="w+")
                bash_file.write(BASH_FORMAT.format(args.seed, trial_commands[current_bash_trial - 1]))
                bash_file.close()

                bash_eval_filename: str = f"{args.output_filename}_eval_{numeric_string_segment}.sh"
                bash_eval_file: TextIO = open(bash_eval_filename, encoding="utf-8", mode="w+")
                bash_eval_file.write(BASH_FORMAT.format(args.seed, trial_eval_commands[current_bash_trial - 1]))
                bash_eval_file.close()

                current_bash_trial += 1
        else:
            raise ValueError(f"Output format <{args.output_format}> not recognized.")
