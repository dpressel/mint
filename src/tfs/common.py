import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, Callable
import math
import logging

logger = logging.getLogger('tfs')


class WeightTiedVocabProjection(nn.Module):
    """Projection layer tied to the input embeddings

    This is equivalent to an nn.Linear(hidden_size, vocab_size, bias=False) where the weights come from the
    input word embeddings.  The embeddings are passed in, and we use their weights for our forward function.
    """

    def __init__(self, from_module: nn.Module, pre_scale=1.0):
        """This uses another module (usually an `nn.Embedding`) to implement its forward function

        :param from_module: Typically an `nn.Embedding` whose weights we use to implement our linear projection
        """
        super().__init__()
        self.from_module = from_module
        self.pre_scale = pre_scale

    @property
    def weight(self):
        return self.from_module.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a dense hidden vector to the vocab space

        :param x: A dense hidden vector
        :return: The vocab space output
        """
        return nn.functional.linear(x * self.pre_scale, self.weight)


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention implementation using scaled dot product

    Converts the input tensor to 3 low-order projections, query, key and value and performs
    multi-headed scaled dot-product attention on them following the Vaswani paper.  The result
    is re-projected a single output representation

    """

    def __init__(self, hidden_size: int, num_heads: int):
        """Each block has the same hidden unit size (`d_model` in the paper).  Must be a multiple of num heads

        :param hidden_size: The number of units (both input and output) of the MHA block
        :param num_heads: The number of heads to split into
        """
        super().__init__()

        d_k = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, num_heads * d_k)
        self.key = nn.Linear(hidden_size, num_heads * d_k)
        self.value = nn.Linear(hidden_size, num_heads * d_k)
        self.output = nn.Linear(num_heads * d_k, hidden_size)
        self.num_heads = num_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param x: A `[B, T, C]` tensor where B is batch, T is time, C is hidden size
        :param mask: An optional mask to apply to the attention matrix
        :return: The attended value vector projected into the output space
        """
        B, T, _ = x.shape
        query_vec = self.query(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        key_vec = self.key(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        value_vec = self.value(x).view(B, T, self.num_heads, -1).transpose(1, 2)

        # [B, H, T_q, D] x [B, H, D, T_k] = [B, H, T_q, T_k]
        dot_prod = (query_vec @ key_vec.transpose(-1, -2)) * self.scale

        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == False, -1e9)

        attn = nn.functional.softmax(dot_prod, dim=-1)
        pre_output = attn @ value_vec

        pre_output = pre_output.transpose(1, 2).contiguous()
        output = self.output(pre_output.view(B, T, -1))
        return output


class MultiHeadedEncoderDecoderAttention(nn.Module):
    """Multi-headed encoder-decoder attention implementation using scaled dot product

    Converts the input tensors to 3 low-order projections, query, key and value and performs
    multi-headed scaled dot-product attention on them following the Vaswani paper.  The result
    is re-projected a single output representation

    """

    def __init__(self, hidden_size: int, num_heads: int):
        """Each block has the same hidden unit size (`d_model` in the paper).  Must be a multiple of num heads

        :param hidden_size: The number of units (both input and output) of the MHA block
        :param num_heads: The number of heads to split into
        """
        super().__init__()

        d_k = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, num_heads * d_k)
        self.key = nn.Linear(hidden_size, num_heads * d_k)
        self.value = nn.Linear(hidden_size, num_heads * d_k)
        self.output = nn.Linear(num_heads * d_k, hidden_size)
        self.num_heads = num_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param src: A `[B, T_k, C]` tensor where B is batch, T_k is time, C is hidden size
        :param dst: A `[B, T_q, C]` tensor where B is batch, T_q is time, C is hidden size
        :param mask: An optional mask to apply to the src (keys) tensor
        :return: The attended value vector projected into the output space
        """
        B, T_k, _ = src.shape
        T_q = dst.shape[1]
        query_vec = self.query(dst).view(B, T_q, self.num_heads, -1).transpose(1, 2)
        key_vec = self.key(src).view(B, T_k, self.num_heads, -1).transpose(1, 2)
        value_vec = self.value(src).view(B, T_k, self.num_heads, -1).transpose(1, 2)

        # [B, H, T_q, D] x [B, H, D, T_k] = [B, H, T_q, T_k]
        dot_prod = (query_vec @ key_vec.transpose(-1, -2)) * self.scale

        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == False, -1e9)

        attn = nn.functional.softmax(dot_prod, dim=-1)
        pre_output = attn @ value_vec

        pre_output = pre_output.transpose(1, 2).contiguous()
        output = self.output(pre_output.view(B, T_q, -1))
        return output


def create_feed_forward_layer(
    hidden_size: int, feed_forward_size: Optional[int] = None, activation: nn.Module = nn.GELU()
):
    """Create a feed-forward layer (called FFN in the paper)

    This uses nn.Sequential to string together each part (the MLP and down-projection back to the output size)

    :param hidden_size: The transformer block size (d_model in the paper)
    :param feed_forward_size: The feed-forward layer size, or 4 * hidden_size.
    :param activation: The activation function, defaults to `nn.GELU()`
    :return: An n.Sequential that wraps the whole FFN transformation block
    """
    d_ff = feed_forward_size if feed_forward_size else 4 * hidden_size
    return nn.Sequential(nn.Linear(hidden_size, d_ff), activation, nn.Linear(d_ff, hidden_size))


class DefaultLayerFactory:
    """Implements Transformer primitives using the basic defaults we have used so far"""

    _instance = None

    @staticmethod
    def get_instance():
        """Access the abstract factory pattern in this way

        It will be created on first use
        """
        if DefaultLayerFactory._instance is None:
            DefaultLayerFactory()

        return DefaultLayerFactory._instance

    def __init__(self):
        if DefaultLayerFactory._instance is not None:
            raise Exception("Singleton constructor call.  Expected no definition")
        self.encoder_multihead_attention = MultiHeadedAttention
        self.decoder_multihead_attention = MultiHeadedAttention
        self.encoder_decoder_attention = MultiHeadedEncoderDecoderAttention
        self.layer_norm = nn.LayerNorm
        self.feed_forward = create_feed_forward_layer
        DefaultLayerFactory._instance = self
