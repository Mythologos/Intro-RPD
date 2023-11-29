from copy import deepcopy
from typing import Callable, Optional

from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, MultiheadAttention
from torch.nn.functional import relu
from torch.nn.modules.transformer import _get_activation_fn


class PreNormTransformerEncoder(Module):
    def __init__(self, input_size: int, hidden_size: int, heads: int, layers: int, activation_function: Callable = relu,
                 dropout: float = 0.1, epsilon: float = 1e-5):
        super().__init__()

        encoder_layer: PreNormTransformerEncoderLayer = \
            PreNormTransformerEncoderLayer(input_size, hidden_size, heads, activation_function, dropout)
        self.encoder = ModuleList([deepcopy(encoder_layer) for _ in range(0, layers)])
        self.final_norm = LayerNorm(input_size, eps=epsilon)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        encoded_source: Tensor = src
        for layer in self.encoder:
            encoded_source = layer(encoded_source, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Following the traditional setup of PreNorm, we include a final LayerNorm.
        encoded_source = self.final_norm(encoded_source)
        return encoded_source


# The below was taken from PyTorch 1.12 and modified.
class PreNormTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        input_size: the number of expected features in the input (required).
        hidden_size: the dimensionality of the Transformer's hidden space (required).
        heads: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation_function: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        epsilon: the eps value in layer normalization components (default=1e-5).
    """
    def __init__(self, input_size: int, hidden_size: int, heads: int, activation_function: Callable,
                 dropout: float = 0.1, epsilon: float = 1e-5):
        super(PreNormTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(input_size, heads, dropout=dropout, batch_first=True)

        # Implementation of feedforward model.
        self.linear1 = Linear(input_size, hidden_size)
        self.linear_dropout = Dropout(dropout)
        self.linear2 = Linear(hidden_size, input_size)

        self.self_attention_norm = LayerNorm(input_size, eps=epsilon)
        self.feedforward_norm = LayerNorm(input_size, eps=epsilon)
        self.self_attention_dropout = Dropout(dropout)
        self.feedforward_dropout = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation_function, str):
            self.activation = _get_activation_fn(activation_function)
        else:
            self.activation = activation_function

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = relu
        super(PreNormTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) \
            -> Tensor:
        r"""
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = x + self._sa_block(self.self_attention_norm(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.feedforward_norm(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x2 = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.self_attention_dropout(x2)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x2 = self.linear2(self.linear_dropout(self.activation(self.linear1(x))))
        return self.feedforward_dropout(x2)
