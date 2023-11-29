from abc import abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module


class EmbeddingLayer(Module):
    def __init__(self, vocabularies: dict[str, dict], embedding_size: int, lemmatizer: Optional = None):
        super().__init__()
        self.vocabularies: dict[str, dict] = vocabularies
        self.lemmatizer: Optional = lemmatizer
        self.embedding_size = embedding_size

    @abstractmethod
    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def prepare_word_sequence(self, chunk: list[str], **kwargs) -> tuple[Tensor, dict]:
        raise NotImplementedError
