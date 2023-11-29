from torch import int64, tensor, Tensor
from torch.nn import Linear, Module

from utils.data.tags import BIOTag
from utils.layers.encoders.encoder import EncoderLayer
from utils.layers.embeddings.embedding import EmbeddingLayer
from utils.layers.modules.blender import Blender
from utils.layers.modules.conditional_random_field import CRF


# The below code is based on that of Robert Guthrie, as presented at the following locations:
#   (1) https://github.com/rguthrie3/DeepLearningForNLPInPytorch
#   (2) https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html


class EncoderCRF(Module):
    def __init__(self, components: dict[str, str], vocabularies: dict[str, dict], embedding: EmbeddingLayer,
                 blender: Blender, encoder: EncoderLayer, linear_input_size: int):
        super().__init__()
        self.components: dict[str, str] = components
        self.vocabularies: dict[str, dict] = vocabularies

        self.embedding: EmbeddingLayer = embedding
        self.blender: Blender = blender
        self.encoder: EncoderLayer = encoder
        self.hidden2tag: Linear = Linear(linear_input_size, len(vocabularies["tags_to_indices"]))
        self.crf: CRF = CRF(vocabularies["tags_to_indices"])

    def forward(self, chunk: Tensor, **kwargs):
        tag_features: Tensor = self.get_tag_features(chunk, **kwargs)

        # Find the best path, given the features.
        score, tag_sequence = self.crf.viterbi_decode(tag_features)
        return score, tag_sequence

    def calculate_nll(self, chunk: Tensor, tags: Tensor, **kwargs):
        chunk_features: Tensor = self.get_tag_features(chunk, **kwargs)
        forward_score = self.crf.compute_forward(chunk_features)
        gold_score = self.crf.score_chunk(chunk_features, tags)
        return forward_score - gold_score

    def prepare_tags(self, tags: list[list[str]]) -> Tensor:
        main_stratum, *_ = tags
        tag_indices = [self.vocabularies["tags_to_indices"][tag] for tag in main_stratum]
        return tensor(tag_indices, dtype=int64)

    def revert_tags(self, tags_indices: list[int], stratum_count: int) -> list[list[str]]:
        stratified_tags: list[list[str]] = []
        first_stratum_tags: list[str] = [self.vocabularies["indices_to_tags"][tag_index] for tag_index in tags_indices]
        stratified_tags.append(first_stratum_tags)
        for i in range(0, stratum_count - 1):
            nth_stratum_tags: list[str] = [BIOTag.OUTSIDE.value for _ in first_stratum_tags]
            stratified_tags.append(nth_stratum_tags)

        return stratified_tags

    def get_tag_features(self, chunk: Tensor, **kwargs):
        embeddings: Tensor = self.embedding(chunk)   # (N', [1 | X]) -> (N', E)

        if self.blender is not None:
            embeddings = self.blender(embeddings, kwargs["boundaries"]).unsqueeze(1)  # (N', E) -> (N, 1, E)

        encodings: Tensor = self.encoder(embeddings)   # (N, 1, E) -> (N, H)
        tag_features: Tensor = self.hidden2tag(encodings)   # (N, H) -> (N, T)
        return tag_features

    def prepare_word_sequence(self, chunk: list[str], **kwargs) -> tuple[Tensor, dict]:
        return self.embedding.prepare_word_sequence(chunk, **kwargs)
