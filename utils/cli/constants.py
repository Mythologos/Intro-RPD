from typing import Sequence


ACTIVATION_FUNCTIONS: Sequence[str] = ("relu", "gelu")
DEFAULT_SPLITS: Sequence[str] = ("training", "validation", "optimization", "test")
PRINT_STYLES: Sequence[str] = ("all", "checkpoint", "none")
REPLACEMENT_STRATEGIES: Sequence[str] = ("any", "singleton", "none")
VISUALIZATIONS: Sequence[str] = ('precision', 'recall', 'f1', 'duration', 'loss')
