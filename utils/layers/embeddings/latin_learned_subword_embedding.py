from typing import Optional

from torch import Tensor
from torch.nn import Embedding

from utils.layers.embeddings.latin_subword_embedding import LatinSubwordEmbeddingLayer


class LatinLearnedSubwordEmbedding(LatinSubwordEmbeddingLayer):
    def __init__(self, vocabularies: dict[str, dict], lemmatizer: Optional, **kwargs):
        super().__init__(vocabularies, kwargs["input_size"], lemmatizer, **kwargs)
        self.subword_embeds = Embedding(len(self.vocabularies["subword_to_index"]), kwargs["input_size"])

    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        subword_embeddings: Tensor = self.subword_embeds(chunk)   # (N + K) -> (N + K, E)
        return subword_embeddings
