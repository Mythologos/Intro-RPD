from typing import Optional

from numpy.typing import NDArray
from torch import from_numpy, stack, Tensor

from utils.data.tokens import UNK_TOKEN
from utils.layers.embeddings.embedding import EmbeddingLayer


class LatinWordEmbedding(EmbeddingLayer):
    def __init__(self, vocabularies: dict[str, dict], lemmatizer: Optional, **kwargs):
        super().__init__(vocabularies, kwargs["input_size"], lemmatizer)
        self.frozen_embeddings: bool = kwargs["frozen_embeddings"]

    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        embeddings: Tensor = chunk.view(len(chunk), 1, -1)
        return embeddings

    def prepare_word_sequence(self, chunk: list[str], **kwargs) -> tuple[Tensor, dict]:
        word_tensors: list[Tensor] = []
        for word in chunk:
            if self.lemmatizer is not None:
                _, current_lemma = self.lemmatizer.lemmatize([word])[0]
            else:
                raise ValueError("Lemmatizer not provided for this model variant.")

            current_word_embedding: Optional[NDArray] = self.vocabularies["word_embeddings"].get(current_lemma, None)
            if current_word_embedding is None:
                current_word_embedding: NDArray = self.vocabularies["word_embeddings"][UNK_TOKEN]

            # We copy the word embedding so that it becomes writable and thus can be used by torch.
            word_tensors.append(from_numpy(current_word_embedding.copy()))
            if self.frozen_embeddings is True:
                word_tensors[-1].requires_grad = False

        return stack(word_tensors), {}
