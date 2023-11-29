import matplotlib.pyplot as plt

from typing import Any, Optional

from matplotlib.ticker import MaxNLocator


def visualize_epoch_curve(epoch_values: list[float], value_name: str, plot_directory: Optional[str] = None):
    fig, ax = plt.subplots()
    epochs: list[int] = list(range(1, len(epoch_values) + 1))
    title_metric: str = value_name.title()

    ax.plot(epochs, epoch_values, marker='o')

    plt.title(f"Plot of {title_metric} Across Epochs")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel(title_metric)

    if plot_directory is None:
        plt.show()
    else:
        plt.savefig(f"{plot_directory}/model_{value_name}.png")


def visualize_metric_progression(training_metric_values: list[float], validation_metric_values: list[float],
                                 metric_name: str, plot_directory: Optional[str] = None):
    fig, ax = plt.subplots()

    epochs: list[int] = list(range(1, len(training_metric_values) + 1))
    title_metric: str = metric_name.title() if metric_name != 'per' else metric_name.upper()

    ax.plot(epochs, training_metric_values, color='tab:blue', marker='o', label='Training')
    ax.plot(epochs, validation_metric_values, color='tab:orange', marker='D', linestyle='dashed', label='Validation')

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Epochs")
    plt.ylabel(title_metric)
    plt.legend(title="Datasets")
    plt.title(f"Plot for Training and Validation {title_metric}")

    if plot_directory is None:
        plt.show()
    else:
        plt.savefig(f"{plot_directory}/model_{metric_name}.png")


def visualize_training_outputs(training_run_outputs: dict[str, Any], visualization_types: list[str],
                               visuals_directory_path: str):
    for visualization_type in visualization_types:
        if visualization_type == "precision":
            visualize_metric_progression(
                training_run_outputs["training_precisions"], training_run_outputs["dev_precisions"],
                visualization_type, visuals_directory_path
            )
        elif visualization_type == "recall":
            visualize_metric_progression(
                training_run_outputs["training_recalls"], training_run_outputs["dev_recalls"], visualization_type,
                visuals_directory_path
            )
        elif visualization_type == "f1":
            visualize_metric_progression(
                training_run_outputs["training_f1s"], training_run_outputs["dev_f1s"], visualization_type,
                visuals_directory_path
            )
        elif visualization_type == "duration":
            visualize_epoch_curve(training_run_outputs["epoch_durations"], visualization_type, visuals_directory_path)
        else:  # visualization_type == "loss"
            visualize_epoch_curve(training_run_outputs["epoch_losses"], visualization_type, visuals_directory_path)
