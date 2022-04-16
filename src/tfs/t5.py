import torch
import torch.nn as nn
import numpy as np
import os
import math
from typing import Optional
from tfs.preln import PreLayerNormTransformerEncoderDecoder, PreLayerNormTransformerSequenceGenerator
import logging
import random

logger = logging.getLogger('tfs')



class MultiHeadedEncoderDecoderAttentionWithoutBias(nn.Module):
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
        self.query = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.key = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.value = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.output = nn.Linear(num_heads * d_k, hidden_size, bias=False)
        self.num_heads = num_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        :param src: A `[B, T_q, C]` tensor where B is batch, T_q is time, C is hidden size
        :param dst: A `[B, T_k, C]` tensor where B is batch, T_k is time, C is hidden size
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


class MultiHeadedRelativeAttentionBias:
    def __init__(self, hidden_size: int, num_heads: int, is_bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128):
        """Each block has the same hidden unit size (`d_model` in the paper).  Must be a multiple of num heads

        :param hidden_size: The number of units (both input and output) of the MHA block
        :param num_heads: The number of heads to split into
        :param is_bidirectional: This will be True for the encoder and false for the decoder

        """
        super().__init__()

        d_k = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.key = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.value = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.output = nn.Linear(num_heads * d_k, hidden_size, bias=False)
        self.num_heads = num_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(d_k)
        self.is_bidirectional = is_bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        rel_embedding = torch.nn.init.kaiming_normal_(torch.empty((self.num_heads, self.num_buckets),
                                                                  dtype=torch.float), nonlinearity='linear')
        self.rel_embedding = nn.Parameter(rel_embedding, requires_grad=True)

    def _relative_position_bucket(self, relative_position):
        """Taken from https://github.com/tensorflow/mesh/blob/bbb6ce7917e2a8ef1f3dc6990fcacd4f3b075acd/mesh_tensorflow/transformer/transformer_layers.py#L1014
        """
        ret = 0
        n = -relative_position
        num_buckets = self.num_buckets
        if self.is_bidirectional:
            num_buckets //= 2
            ret += torch.lt(n, 0).to(dtype=torch.long) * num_buckets
            n = torch.abs(n).to(dtype=torch.long)
        else:
            n = torch.maximum(n, 0).to(dtype=torch.long)

        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + (
                torch.log(n.to(dtype=torch.float32) / max_exact)
                / math.log(self.max_distance / max_exact) * (num_buckets - max_exact)).to(dtype=torch.long)
        val_if_large = torch.minimum(val_if_large, torch.tensor(num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

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

        memory_position = torch.arange(T).view(1, -1)
        query_position = torch.arange(T).view(-1, 1)
        relative_position = memory_position - query_position

        # [B, H, T_q, D] x [B, H, D, T_k] = [B, H, T_q, T_k]
        dot_prod = (query_vec @ key_vec.transpose(-1, -2)) * self.scale

        rp_bucket = self._relative_position_bucket(relative_position)
        relative_attention_bias = self.rel_embedding[:, rp_bucket]
        dot_prod += relative_attention_bias

        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == False, -1e9)

        attn = nn.functional.softmax(dot_prod, dim=-1)
        pre_output = attn @ value_vec

        pre_output = pre_output.transpose(1, 2).contiguous()
        output = self.output(pre_output.view(B, T, -1))
        return output


class LayerNormWithoutAdditiveBias(nn.Module):
    """T5 uses a layer norm with no additive bias

    """
    def __init__(self, hidden_dim: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    @classmethod
    def from_layer_norm(cls, module):
        ln = cls(module.weight.shape[0], module.eps)
        ln.weight = module.weight
        return ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu)**2).mean(-1, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mu)/std
        return y * self.weight


class WordOnlyEmbedding(nn.Module):
    """Embeddings for T5

    The embeddings for T5 are just lookup table embeddings with a bias term provided at each layer.
    No `token_type` is used and the `max_seq_len` is ignored
    """

    def __init__(self, vocab_dim: int, hidden_dim: int = 768, padding_idx: int = 0, max_seq_len: int = 512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx)

    def forward(self, x: torch.Tensor, token_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Takes a tensor of shape `[B, T]` and an optional `token_type` of same shape

        :param x: A tensor of word one-hots, shape `[B, T]`
        :param token_type: Ignored for T5
        :return: The sum of the positional and word embeddings
        """
        embed = self.word_embeddings(x)
        return embed

    @property
    def weight(self):
        """Access word_embeddings weights

        :return: The word_embeddings weights
        """
        return self.word_embeddings.weight


class T5EncoderDecoder(PreLayerNormTransformerEncoderDecoder):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 1,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dropout: float = 0.1,
        layer_norm_eps=1e-12,
        activation: nn.Module = nn.ReLU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 512,
    ):
        super().__init__(
            WordOnlyEmbedding,
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
            max_seq_len,
            MultiHeadedEncoderDecoderAttentionWithoutBias,
            lambda x, y: MultiHeadedRelativeAttentionBias(x, y, True),
            lambda x, y: MultiHeadedRelativeAttentionBias(x, y, False),
            LayerNormWithoutAdditiveBias,
        )


class T5SequenceGenerator(PreLayerNormTransformerSequenceGenerator):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 1,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        dropout: float = 0.1,
        layer_norm_eps=1e-12,
        activation: nn.Module = nn.ReLU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 512,
        **kwargs,
    ):
        super().__init__(
            WordOnlyEmbedding,
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
            max_seq_len,
        )

    def create_loss(self):
        return nn.CrossEntropyLoss(ignore_index=1)

