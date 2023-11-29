from copy import deepcopy
from random import shuffle
from sys import stdout
from time import time
from typing import Any, Callable, TextIO, Union

from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from tqdm import tqdm

from utils.data.loaders.constants import ParallelismDataset
from utils.training.evaluation_loop import evaluate


def train(training_model: Module, training_device: str, dataset: ParallelismDataset,
          optimizer: Callable, optimizer_arguments: dict[str, Any], training_arguments: dict[str, Any],
          evaluation_arguments: dict[str, Any], file_arguments: dict[str, Union[str, TextIO, None]]) -> \
        tuple[Module, dict[str, Any]]:
    outputs: dict[str, Any] = {
        "best_epoch_count": 0,   # This refers to the count of the epoch, starting from 1.
        "best_epoch_index": 0,  # This refers to the actual 0-based index for the epoch.
        "best_f1_score": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_scoring_structure": None,
        "dev_f1s": [],
        "dev_precisions": [],
        "dev_recalls": [],
        "epoch_durations": [],
        "epoch_losses": [],
        "training_f1s": [],
        "training_precisions": [],
        "training_recalls": [],
        "total_loss": 0.0
    }

    best_model: Module = deepcopy(training_model)
    optimizer: Optimizer = optimizer(training_model.parameters(), **optimizer_arguments)

    current_epoch: int = 1
    current_patience: int = 0
    while current_epoch <= training_arguments["epochs"] and current_patience < training_arguments["patience"]:
        epoch_start: float = time()
        if training_arguments["print_style"] != "none":
            print(f"Starting epoch {current_epoch}...", flush=True)

        if file_arguments["training_file"] is not None:
            file_arguments["training_file"].write(f"Starting epoch {current_epoch}...\n")

        training_model.train()

        epoch_loss: float = 0.0
        for unit, tags, _ in tqdm(dataset[training_arguments["training_partition"]], file=stdout,
                                  disable=not training_arguments["tqdm"]):
            # We turn the input sentences and tags into tensors.
            unit_tensor, unit_kwargs = training_model.prepare_word_sequence(unit)
            unit_tensor = unit_tensor.to(training_device)

            tag_tensor: Tensor = training_model.prepare_tags(tags).to(training_device)
            sample_loss: Tensor = training_model.calculate_nll(unit_tensor, tag_tensor, **unit_kwargs)

            # We set our gradient back to zero.
            optimizer.zero_grad()
            # We back-propagate the loss.
            sample_loss.backward()
            # We clip gradients.
            clip_grad_norm_(training_model.parameters(), 1.0)
            # We perform a step with the optimizer.
            optimizer.step()

            epoch_loss += sample_loss.item()

        epoch_end: float = time()

        if training_arguments["print_style"] != "none":
            print(f"Finishing epoch {current_epoch}. Validating...", flush=True)

        if file_arguments["training_file"] is not None:
            file_arguments["training_file"].write(f"Finishing epoch {current_epoch}. Validating...\n")

        outputs["epoch_durations"].append(epoch_end - epoch_start)
        outputs["epoch_losses"].append(epoch_loss)
        outputs["total_loss"] += epoch_loss

        # Next, we evaluate the model on both the training set and the validation set.
        training_results: dict[str, Any] = evaluate(
            training_model, training_device, dataset, evaluation_arguments, file_arguments, "training_file",
            training_arguments["tqdm"], training_arguments["training_partition"]
        )

        outputs["training_precisions"].append(training_results["precision"])
        outputs["training_recalls"].append(training_results["recall"])
        outputs["training_f1s"].append(training_results["f1"])

        validation_results: dict[str, Any] = evaluate(
            training_model, training_device, dataset, evaluation_arguments, file_arguments, "validation_file",
            training_arguments["tqdm"]
        )

        outputs["dev_precisions"].append(validation_results["precision"])
        outputs["dev_recalls"].append(validation_results["recall"])
        outputs["dev_f1s"].append(validation_results["f1"])

        validation_statistics_display: str = validation_results["scoring_structure"].get_statistics_display()
        if outputs["best_f1_score"] == 0.0 or validation_results["f1"] >= outputs["best_f1_score"]:
            if training_arguments["print_style"] != "none":
                print(f"We save a model with the following results:\n{validation_statistics_display}")

            outputs["best_epoch_count"] = current_epoch
            outputs["best_epoch_index"] = current_epoch - 1   # Since current_epoch starts at 1, we subtract 1 to index.
            outputs["best_f1_score"] = validation_results["f1"]
            outputs["best_precision"] = validation_results["precision"]
            outputs["best_recall"] = validation_results["recall"]
            outputs["best_scoring_structure"] = validation_results["scoring_structure"]
            best_model = deepcopy(training_model)

            # We add a conditional here to allow for the "most recent" model to be saved,
            # even if the best F1 score did not change.
            # This also prevents a model from running to its maximum number of epochs if it doesn't pick up at all
            # after a certain amount of time.
            if outputs["best_f1_score"] > 0.0:
                current_patience = 0   # We reset the patience, since the last model did better.
            else:
                current_patience += 1
        else:
            if training_arguments["print_style"] != "none":
                print(f"We do NOT save a model with the following results:\n{validation_statistics_display}")
            current_patience += 1   # We increment the patience.

        # We prepare for the next epoch...
        shuffle(dataset["training"])
        current_epoch += 1
    return best_model, outputs
