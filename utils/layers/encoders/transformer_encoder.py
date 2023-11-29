from typing import Callable

from torch import Tensor

from utils.layers.encoders.encoder import EncoderLayer
from utils.layers.modules.modified_transformer_encoder import PreNormTransformerEncoder
from utils.layers.modules.positional_encodings import PositionalEncoding


class TransformerEncoder(EncoderLayer):
    def __init__(self, input_size: int, hidden_size: int, layers: int, heads: int, activation_function: Callable):
        super().__init__(input_size, hidden_size, layers)
        self.positional_encodings = PositionalEncoding(input_size)
        self.encoder = PreNormTransformerEncoder(input_size, hidden_size, heads, layers, activation_function)

    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:
        positioned_embeddings: Tensor = self.positional_encodings(embeddings)   # (N, 1, E) -> (N, 1, E)
        encodings: Tensor = self.encoder(positioned_embeddings)  # (N, 1, H) -> (N, 1, H)
        encodings = encodings.squeeze(1)   # (N, 1, H) -> (N, H)
        return encodings
