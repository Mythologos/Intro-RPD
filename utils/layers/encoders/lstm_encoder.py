from torch import Tensor
from torch.nn import LSTM, Module

from utils.layers.encoders.encoder import EncoderLayer


class LSTMEncoder(EncoderLayer):
    def __init__(self, input_size: int, hidden_size: int, layers: int):
        super().__init__(input_size, hidden_size, layers)
        self.encoder: Module = LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size // 2, num_layers=self.layers, bidirectional=True
        )

    def forward(self, embeddings: Tensor, **kwargs) -> Tensor:
        encodings, _ = self.encoder(embeddings)
        sequence_length, *_ = encodings.shape
        encodings: Tensor = encodings.view(sequence_length, self.hidden_size)
        return encodings
