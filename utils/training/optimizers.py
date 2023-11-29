from inspect import getfullargspec
from typing import Any

from torch import optim
from torch.optim import Optimizer


def get_optimizer(optimizer_name: str):
    if hasattr(optim, optimizer_name):
        optimizer: Optimizer = getattr(optim, optimizer_name)
        if not isinstance(Optimizer, optimizer.__class__):
            raise ValueError(f"The item selected is in torch.optim, but it is not a valid optimizer.")
    else:
        raise ValueError(f"The optimizer with name <{optimizer_name}> is currently not supported by PyTorch. "
                         f"Please try again.")
    return optimizer


def define_optimizer_args(optimizer: Optimizer, general_kwargs: dict[str, Any]) -> dict[str, Any]:
    current_optimizer_kwargs: dict[str, Any] = {}

    # The below uses function inspection so that only relevant arguments are added to optimizer_args.
    # It requires using consistent nomenclature,
    # and it assumes that PyTorch is standardized enough among its optimizers for this to work.
    # It avoids examining the "self" and "params" arguments, hence the '[2:]'.
    optional_optimizer_parameters: list[str] = getfullargspec(optimizer.__init__).args[2:]
    for parameter in optional_optimizer_parameters:
        if general_kwargs.get(parameter, None) is not None:
            current_optimizer_kwargs[parameter] = general_kwargs[parameter]

    return current_optimizer_kwargs
