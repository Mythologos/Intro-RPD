from random import choices
from typing import Optional

from torch import int64, tensor, Tensor
from torch.nn import Embedding

from utils.data.tokens import UNK_TOKEN
from utils.layers.embeddings.embedding import EmbeddingLayer


REPLACEMENT_VALUES: tuple[bool, bool] = (True, False)


class LearnedWordEmbedding(EmbeddingLayer):
    def __init__(self, vocabularies: dict[str, dict], lemmatizer: Optional, **kwargs):
        super().__init__(vocabularies, kwargs["input_size"], lemmatizer)
        self.learned_embeddings = Embedding(len(vocabularies["words_to_indices"]), kwargs["input_size"])
        self.replacement_probability = kwargs["replacement_probability"]
        self.replacement_weights = (self.replacement_probability, 1 - self.replacement_probability)
        self.replacement_strategy = kwargs["replacement_strategy"]

    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        embeddings: Tensor = self.learned_embeddings(chunk).view(len(chunk), 1, -1)
        return embeddings

    def prepare_word_sequence(self, chunk: list[str], **kwargs) -> tuple[Tensor, dict]:
        word_indices: list[int] = []
        for word in chunk:
            if self.lemmatizer is not None:
                _, processed_word = self.lemmatizer.lemmatize([word])[0]
            else:
                processed_word: str = word

            current_word_index: Optional[int] = self.vocabularies["words_to_indices"].get(processed_word, None)
            if current_word_index is not None:
                current_word_index = self._handle_replacement(processed_word, current_word_index)
                word_indices.append(current_word_index)
            else:
                word_indices.append(self.vocabularies["words_to_indices"][UNK_TOKEN])

        return tensor(word_indices, dtype=int64), {}

    def _handle_replacement(self, processed_word: str, current_word_index: int) -> int:
        if self.training is True:
            is_singleton: bool = self.vocabularies["words_to_frequency"].get(processed_word, 0) <= 1
            if self.replacement_strategy == "any" or \
                    (self.replacement_strategy == "singleton" and is_singleton):
                # With a pre-set probability, we replace the token with <UNK> during training.
                should_replace: bool = choices(REPLACEMENT_VALUES, self.replacement_weights, k=1)[-1]
                if should_replace is True:
                    current_word_index = self.vocabularies["words_to_indices"][UNK_TOKEN]

        return current_word_index
