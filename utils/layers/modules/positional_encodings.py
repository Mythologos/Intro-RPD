from math import log

from torch import arange, cos, exp, sin, zeros, Tensor
from torch.nn import Dropout, Module


MAX_PROGRESSION_CONSTANT: float = 10000.0


class PositionalEncoding(Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = arange(max_len).unsqueeze(1)
        div_term = exp(arange(0, hidden_dim, 2) * ((-1 * log(MAX_PROGRESSION_CONSTANT)) / hidden_dim))
        positional_encodings = zeros(max_len, 1, hidden_dim)
        positional_encodings[:, 0, 0::2] = sin(position * div_term)
        positional_encodings[:, 0, 1::2] = cos(position * div_term)
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.positional_encodings[:x.size(0)]
        return self.dropout(x)
