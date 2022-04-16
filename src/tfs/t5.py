import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
from tfs.preln import PreLayerNormTransformerEncoderDecoder, PreLayerNormTransformerSequenceGenerator
import logging
import random

logger = logging.getLogger('tfs')


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

