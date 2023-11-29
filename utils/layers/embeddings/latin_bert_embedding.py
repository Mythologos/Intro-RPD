from typing import Optional

from torch import cat, Tensor, tensor_split
from transformers import BertConfig, BertModel

from utils.layers.embeddings.latin_subword_embedding import LatinSubwordEmbeddingLayer


MAX_INPUT_SIZE: int = 512


class LatinBertEmbedding(LatinSubwordEmbeddingLayer):
    def __init__(self, vocabularies: dict[str, dict], lemmatizer: Optional, **kwargs):
        bert_config: BertConfig = BertConfig.from_pretrained(kwargs["pretrained_filepath"])
        super().__init__(vocabularies, bert_config.hidden_size, lemmatizer, **kwargs)

        self.bert = BertModel.from_pretrained(kwargs["pretrained_filepath"])
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
