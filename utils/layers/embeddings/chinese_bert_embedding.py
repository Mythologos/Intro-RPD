from typing import Any, Optional

from torch import cat, int64, tensor, tensor_split, Tensor
from transformers import BertConfig, BertModel, BertTokenizer

from utils.data.tokens import BERTSpecialToken
from utils.layers.embeddings.embedding import EmbeddingLayer


MAX_INPUT_SIZE: int = 512


class ChineseBertEmbedding(EmbeddingLayer):
    def __init__(self, vocabularies: dict[str, dict], lemmatizer: Optional, **kwargs):
        bert_config: BertConfig = BertConfig.from_pretrained(kwargs["pretrained_filepath"])
        super().__init__(vocabularies, bert_config.hidden_size, lemmatizer)

        self.bert = BertModel.from_pretrained(kwargs["pretrained_filepath"])
        self.tokenizer = BertTokenizer.from_pretrained(kwargs["tokenizer_filepath"])
        self.frozen_embeddings: bool = kwargs["frozen_embeddings"]

        if self.frozen_embeddings is True:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def forward(self, chunk: Tensor, **kwargs) -> Tensor:
        # We obtain segments of BERT, overcoming sequence length issues.
        initial_embeddings: Tensor = chunk.view(1, -1)   # (N + K) -> (1, N + K)
        # We chunk the embeds into a size which BERT can handle; that is, into chunks of 512.
        encoding_segments: list[int] = [i for i in range(MAX_INPUT_SIZE, len(chunk), MAX_INPUT_SIZE)]
        embed_segments: list[Tensor] = tensor_split(initial_embeddings, encoding_segments, dim=-1)
        individual_bert_embeddings: list[Tensor] = []
        for i in range(0, len(embed_segments)):
            bert_embedding: Tensor = self.bert(input_ids=embed_segments[i]).last_hidden_state
            individual_bert_embeddings.append(bert_embedding.squeeze(0))

        # We combine the collected BERT segments to get a full representation of the sequence.
        bert_embeddings = cat(individual_bert_embeddings, dim=0)   # list[(N' + K', D)] -> (N + K, D)
        return bert_embeddings

    def prepare_word_sequence(self, chunk: list[str], **kwargs) -> tuple[Tensor, dict]:
        if self.lemmatizer is not None:
            processed_chunk = [lemma for (_, lemma) in self.lemmatizer.lemmatize(chunk)]
        else:
            processed_chunk = chunk

        tokens: list[list[str]] = [self.tokenizer.tokenize(word) for word in processed_chunk]
        tokens.insert(0, [BERTSpecialToken.CLASS_TOKEN.token])
        tokens.append([BERTSpecialToken.SEPARATION_TOKEN.token])
        nested_subword_indices: list[list[int]] = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]

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
