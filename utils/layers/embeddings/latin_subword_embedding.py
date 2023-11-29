from abc import abstractmethod
from typing import Any, Optional

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from torch import int64, tensor, Tensor

from utils.data.tokens import STARTING_VOCABULARY, BERTSpecialToken
from utils.layers.embeddings.embedding import EmbeddingLayer


class LatinSubwordEmbeddingLayer(EmbeddingLayer):
    def __init__(self, vocabularies: dict[str, dict], embedding_size: int, lemmatizer: Optional, **kwargs):
        super().__init__(vocabularies, embedding_size, lemmatizer)
        self.subword_tokenizer = self._handle_subword_tokenizer(kwargs["tokenizer_filepath"])
        self.vocabularies["subword_to_index"] = self.subword_tokenizer._subtoken_string_to_id

    @abstractmethod
    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _handle_subword_tokenizer(tokenizer_filepath: str) -> SubwordTextEncoder:
        subword_tokenizer: SubwordTextEncoder = SubwordTextEncoder(tokenizer_filepath)

        special_token_count: int = len([token for token in BERTSpecialToken])   # type: ignore
        for key, value in subword_tokenizer._subtoken_string_to_id.items():
            subword_tokenizer._subtoken_string_to_id[key] = value + special_token_count

        for token, value in STARTING_VOCABULARY.items():
            subword_tokenizer._subtoken_string_to_id[token] = value

        return subword_tokenizer

    def prepare_word_sequence(self, chunk: list[str], **kwargs) -> tuple[Tensor, dict]:
        if self.lemmatizer is not None:
            processed_chunk = [lemma for (_, lemma) in self.lemmatizer.lemmatize(chunk)]
        else:
            processed_chunk = chunk

        nested_subword_indices: list[list[int]] = [self.subword_tokenizer.encode(word) for word in processed_chunk]
        nested_subword_indices.insert(0, [self.vocabularies["subword_to_index"][BERTSpecialToken.CLASS_TOKEN.token]])
        nested_subword_indices.append([self.vocabularies["subword_to_index"][BERTSpecialToken.SEPARATION_TOKEN.token]])
        subword_indices: list[int] = []
        boundaries: list[int] = [0]

        for nest_number, nest in enumerate(nested_subword_indices, 1):
            next_bound: int = boundaries[-1] + len(nest)
            subword_indices.extend(nest)
            if nest_number != len(nested_subword_indices):
                boundaries.append(next_bound)  # We avoid bounding the [SEP] token.

        boundaries.pop(0)  # We get rid of the initial "0", since we don't want to keep [CLS].

        chunk_kwargs: dict[str, Any] = {"boundaries": boundaries}
        return tensor(subword_indices, dtype=int64), chunk_kwargs
