from abc import abstractmethod

from torch import Tensor
from torch.nn import Module


class EncoderLayer(Module):
    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = layers

    @abstractmethod
    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError
