import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional, Callable
import math
from common import DefaultLayerFactory, WeightTiedVocabProjection
import logging

logger = logging.getLogger('tfs')



class TransformerEncoderLayer(nn.Module):
    """A single (post-layer-norm style) Transformer layer

    This layer implements a post-layer-norm style Transformer.  The MultiHeadedAttention is applied first, with
    optional dropout and added to its input, followed by normalization.  Then the FFN block is applied, where
    an MLP layer with a larger size is applied followed by an activation and down-projection back to the input size.
    Dropout is again applied, and again we add the output to the input of FFN, followed by a layer norm.

    As this is a post-layer-norm architecture, a normalization operation should be applied prior to sending the
    data through this layer

    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        layer_factory = None,
    ):
        """Initialize our transformer, uses bert-base defaults

        :param hidden_size: Size of the transformer inputs and outputs (d_model in the paper)
        :param num_heads: The number of heads for multi-headed attention
        :param dropout: A dropout to apply to each sub-blocks outputs
        :param layer_norm_eps: The noise applied in the layer norm calculation
        :param activation: The activation function to use
        :param feed_forward_size:  The optional size of the FFN internal representation.  Defaults to 4*hidden_size
        :param layer_factory: An optional implementation of all layers, useful for specific model implementation details
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_ff = feed_forward_size
        self.self_attention = layer_factory.encoder_multihead_attention(hidden_size, num_heads)
        self.self_attention_layer_norm = layer_factory.layer_norm(hidden_size, layer_norm_eps)
        self.ffn = layer_factory.feed_forward(hidden_size, feed_forward_size, activation)
        self.output_layer_norm = layer_factory.layer_norm(hidden_size, layer_norm_eps)

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout operator in graph only if training

        TODO: this function could also test dropout to make sure its > 0, pruning an unnecessary op
        if training with no dropout

        :param x: The output of the sub-layer
        :return: A (maybe) dropped out version of the input
        """
        return nn.functional.dropout(x, self.dropout) if self.training else x

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Pass an x tensor and optional mask through the transformer layer

        :param x: A `[B, T, C]` tensor where B is batch, T is time, and C is the num hidden units
        :param mask: An optional attention mask.  True where the input is valid, and false where it isnt
        :return: The output of the block
        """
        y = self.self_attention_layer_norm(x + self.maybe_dropout(self.self_attention(x, mask)))
        y = self.output_layer_norm(y + self.maybe_dropout(self.ffn(y)))
        return y


class TransformerEncoder(nn.Module):
    """A Post-Layer Norm Transformer Encoder (with no task heads)

    This encoder encapsulates the entire front-end of the Transformer from one-hots up to the final
    encoding.  For tasks like MLM and fine-tuning we will inherit this module and provide additional
    functionality to the forward.

    This set up via inheritance to keep sub-classing and configuration params being passed to a minimum
    The tutorial mentions other ways that you could organize this

    """

    def __init__(
        self,
        EmbeddingClass: Callable,
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        layer_norm_eps=1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 512,
        do_embeddings_layer_norm=True,
        layer_factory=None,
    ):
        """Set up initialization for a (post-layer-norm) Transformer.  Defaults to bert-base settings

        :param vocab_size: The size of the input vocabulary
        :param padding_idx: The padding index, defaults to 0
        :param hidden_size: The number of hidden units
        :param num_heads: The number of heads for multi-headed attn.  Should divide evenly into hidden_size
        :param num_layers: The number of transformer layers (MHA+FFN) in the architecture
        :param dropout: The value to apply for dropout
        :param layer_norm_eps: The noising term for layer norm
        :param activation: The activation function to use throughout
        :param feed_forward_size: An optional value to set for the FFN MLP output size, defaults to 4*hidden_size
        :param layer_factory: An optional implementation of all layers, useful for specific model implementation details

        """
        super().__init__()
        self.padding_idx = padding_idx

        if layer_factory is None:
            layer_factory = DefaultLayerFactory.get_instance()

        self.embeddings_layer_norm = (
            layer_factory.layer_norm(hidden_size, layer_norm_eps) if do_embeddings_layer_norm else nn.Identity()
        )
        self.embeddings = EmbeddingClass(vocab_size, hidden_size, padding_idx=padding_idx, max_seq_len=max_seq_len)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size, layer_factory)
                for _ in range(num_layers)
            ]
        )
        self.LayerNormImpl = layer_factory.layer_norm

    @property
    def hidden_size(self):
        """Useful to see the hidden size of the arch., but we dont a member var, its going to be all over the layers
        :return:
        """
        return self.embeddings.word_embeddings.weight.shape[1]

    @property
    def vocab_size(self):
        """Useful to see the vocab size, but we dont need to store as a member, its the first dim of word embeddings

        :return:
        """
        return self.embeddings.word_embeddings.weight.shape[0]

    def create_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """For input padded using the padding_idx, generate an attention mask for that

        :param x:
        :return:
        """
        mask = x != self.padding_idx
        return mask.unsqueeze(1).unsqueeze(1).to(device=x.device)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        :param x: A one-hot (long) tensor of shape `[B, T]`
        :param mask: An optional mask to take in for attention
        :param token_type:
        :return:
        """
        y = self.embeddings(x, token_type)
        y = self.embeddings_layer_norm(y)
        for t in self.encoder:
            y = t(y, mask)
        return y

    def init_layer_weights(self, module):
        """This not directly used on initialization.  If you want to use it, call `module.apply()` on it

        The base classes do make use of it for MLM and pooling in their constructors
        :param module:
        :return:
        """
        if isinstance(module, (nn.Linear, nn.Embedding, self.LayerNormImpl)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, self.LayerNormImpl)) and module.bias is not None:
            module.bias.data.zero_()


class TransformerDecoderLayer(nn.Module):
    """A single (post-layer-norm style) Transformer Decoder layer

    This layer implements a post-layer-norm style Transformer Decoder (in the NMT/Encoder-Decoder sense).
    This module contains both self-attention (used in the decoder portion, and Encoder-Decoder cross-attention)

    As this is a post-layer-norm architecture, a normalization operation should be applied prior to sending the
    data through this layer

    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        layer_factory = None,
    ):
        """Initialize our transformer, uses bert-base defaults

        :param hidden_size: Size of the transformer inputs and outputs (d_model in the paper)
        :param num_heads: The number of heads for multi-headed attention
        :param dropout: A dropout to apply to each sub-blocks outputs
        :param layer_norm_eps: The noise applied in the layer norm calculation
        :param activation: The activation function to use
        :param feed_forward_size:  The optional size of the FFN internal representation.  Defaults to 4*hidden_size
        :param layer_factory: An optional implementation of all layers, useful for specific model implementation details

        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_ff = feed_forward_size
        if layer_factory is None:
            layer_factory = DefaultLayerFactory.get_instance()
        self.self_attention = layer_factory.decoder_multihead_attention(hidden_size, num_heads)
        self.self_attention_layer_norm = layer_factory.layer_norm(hidden_size, layer_norm_eps)
        self.encoder_attention = layer_factory.encoder_decoder_attention(hidden_size, num_heads)
        self.encoder_attention_layer_norm = layer_factory.layer_norm(hidden_size, layer_norm_eps)
        self.ffn = layer_factory.feed_forward(hidden_size, feed_forward_size, activation)
        self.output_layer_norm = layer_factory.layer_norm(hidden_size, layer_norm_eps)

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout operator in graph only if training

        TODO: this function could also test dropout to make sure its > 0, pruning an unnecessary op
        if training with no dropout

        :param x: The output of the sub-layer
        :return: A (maybe) dropped out version of the input
        """
        return nn.functional.dropout(x, self.dropout) if self.training else x

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        dst_mask: Optional[torch.Tensor] = None,
    ):
        """Pass an x tensor and optional mask through the transformer layer

        :param x: A `[B, T, C]` tensor where B is batch, T is time, and C is the num hidden units
        :param mask: An optional attention mask.  True where the input is valid, and false where it isnt
        :return: The output of the block
        """
        y = self.self_attention_layer_norm(dst + self.maybe_dropout(self.self_attention(dst, dst_mask)))
        y = self.encoder_attention_layer_norm(y + self.maybe_dropout(self.encoder_attention(src, y, src_mask)))
        y = self.output_layer_norm(y + self.maybe_dropout(self.ffn(y)))
        return y


class TransformerEncoderDecoder(nn.Module):
    """A Post-Layer Norm Transformer Decoder (with no task heads)

    This encoder encapsulates the entire front-end of the Transformer from one-hots up to the final
    encoding.  For tasks like MLM and fine-tuning we will inherit this module and provide additional
    functionality to the forward.

    This set up via inheritance to keep sub-classing and configuration params being passed to a minimum
    The tutorial mentions other ways that you could organize this

    """

    def __init__(
        self,
        EmbeddingClass: Callable,
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        layer_norm_eps=1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 512,
        do_embeddings_layer_norm=True,
        layer_factory = None,
    ):
        """Set up initialization for a (post-layer-norm) Transformer.  Defaults to bert-base settings

        :param vocab_size: The size of the input vocabulary
        :param padding_idx: The padding index, defaults to 0
        :param hidden_size: The number of hidden units
        :param num_heads: The number of heads for multi-headed attn.  Should divide evenly into hidden_size
        :param num_layers: The number of transformer layers (MHA+FFN) in the architecture
        :param dropout: The value to apply for dropout
        :param layer_norm_eps: The noising term for layer norm
        :param activation: The activation function to use throughout
        :param feed_forward_size: An optional value to set for the FFN MLP output size, defaults to 4*hidden_size
        :param layer_factory: An optional implementation of all layers, useful for specific model implementation details
        """
        super().__init__()
        self.padding_idx = padding_idx
        if layer_factory is None:
            layer_factory = DefaultLayerFactory.get_instance()
        self.encoder_embeddings_layer_norm = (
            layer_factory.layer_norm(hidden_size, layer_norm_eps) if do_embeddings_layer_norm else nn.Identity()
        )
        self.decoder_embeddings_layer_norm = (
            layer_factory.layer_norm(hidden_size, layer_norm_eps) if do_embeddings_layer_norm else nn.Identity()
        )
        self.encoder_embeddings = EmbeddingClass(
            vocab_size, hidden_size, padding_idx=padding_idx, max_seq_len=max_seq_len
        )
        self.decoder_embeddings = EmbeddingClass(
            vocab_size, hidden_size, padding_idx=padding_idx, max_seq_len=max_seq_len
        )

        self.decoder_embeddings.word_embeddings = self.encoder_embeddings.word_embeddings

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size, layer_factory)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size, layer_factory)
                for _ in range(num_decoder_layers)
            ]
        )

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    (
                        max_seq_len,
                        max_seq_len,
                    ),
                    dtype=torch.uint8,
                )
            )
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self.LayerNormImpl = layer_factory.layer_norm

    @property
    def hidden_size(self):
        """Useful to see the hidden size of the arch., but we dont a member var, its going to be all over the layers
        :return:
        """
        return self.encoder_embeddings.word_embeddings.weight.shape[1]

    @property
    def vocab_size(self):
        """Useful to see the vocab size, but we dont need to store as a member, its the first dim of word embeddings

        :return:
        """
        return self.encoder_embeddings.word_embeddings.weight.shape[0]

    def create_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """For input padded using the padding_idx, generate an attention mask for that

        :param x:
        :return:
        """
        mask = x != self.padding_idx
        return mask.unsqueeze(1).unsqueeze(1).to(device=x.device)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        dst_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        :param src: A one-hot (long) tensor of shape `[B, T_k]`
        :param dst: A one-hot (long) tensor of shape `[B, T_q]`
        :param src_mask: An optional mask to take in for attention
        :param dst_mask: An optional mask to take in for attention
        :return:
        """

        src_enc = self.encode(src, src_mask)
        dst_enc = self.decode(src_enc, dst, src_mask, dst_mask)
        return dst_enc

    def decode(self, src_enc, dst, src_mask: Optional[torch.Tensor] = None, dst_mask: Optional[torch.Tensor] = None):
        futures_mask = self.causal_mask[:, :, : dst.shape[1], : dst.shape[1]]
        if dst_mask is not None:
            futures_mask = dst_mask & futures_mask.to(dtype=torch.bool)
        dst_enc = self.decoder_embeddings(dst)
        dst_enc = self.decoder_embeddings_layer_norm(dst_enc)
        for t in self.decoder:
            dst_enc = t(src_enc, dst_enc, src_mask, futures_mask)
        return dst_enc

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_enc = self.encoder_embeddings(src)
        src_enc = self.encoder_embeddings_layer_norm(src_enc)
        for t in self.encoder:
            src_enc = t(src_enc, src_mask)
        return src_enc

    def init_layer_weights(self, module):
        """This not directly used on initialization.  If you want to use it, call `module.apply()` on it

        The base classes do make use of it for MLM and pooling in their constructors
        :param module:
        :return:
        """
        if isinstance(module, (nn.Linear, nn.Embedding, self.LayerNormImpl)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, self.LayerNormImpl)) and module.bias is not None:
            module.bias.data.zero_()


class TransformerSequenceGenerator(TransformerEncoderDecoder):
    """An encoder-decoder that produces word output in the decoder

    For training, this works with teacher forcing, where both the encoder inputs and the
    lagged generated tokens at each timestep are provided, starting with some well-known
    decoder begin token.

    At inference time, we will do some decoding over time, and so we need to be able to
    call the encoder once, and the decoder N times
    """
    def __init__(
        self,
        EmbeddingClass: Callable,
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        layer_norm_eps=1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 1024,
        do_embeddings_layer_norm=True,
        layer_factory = None,

    ):
        super().__init__(
            EmbeddingClass,
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
            do_embeddings_layer_norm,
            layer_factory,
        )
        self.output_proj = WeightTiedVocabProjection(self.decoder_embeddings.word_embeddings)
        self.apply(self.init_layer_weights)

    def decode(self, src_enc, dst, src_mask: Optional[torch.Tensor] = None, dst_mask: Optional[torch.Tensor] = None):
        dst_enc = super().decode(src_enc, dst, src_mask, dst_mask)
        return self.output_proj(dst_enc)






