from torch import Tensor
from torch.nn import Identity

from utils.layers.encoders.encoder import EncoderLayer


class IdentityEncoder(EncoderLayer):
    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super().__init__(input_size, hidden_size, layers)
        self.identity_function = Identity()

    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:
        encodings: Tensor = self.identity_function(embeddings)
        return encodings.squeeze(1)
