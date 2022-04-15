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

    def __init__(self, from_module):
        super().__init__()
        self.from_module = from_module

    @property
    def weight(self):
        return self.from_module.weight

    def forward(self, x):
        return nn.functional.linear(x, self.weight)


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


def create_feed_forward_layer(
    hidden_size: int, feed_forward_size: Optional[int] = None, activation: nn.Module = nn.GELU()
):
    """Create a feed-forward layer (called FFN in the paper)

    This uses nn.Sequential to string together each part (the MLP and down-projection back to the output size)

    :param hidden_size: The transformer block size (d_model in the paper)
    :param feed_forward_size: The feed-forward layer size, or 4 * hidden_size.
    :param activation: The activation function, defaults to GELU
    :return: An n.Sequential that wraps the whole FFN transformation block
    """
    d_ff = feed_forward_size if feed_forward_size else 4 * hidden_size
    return nn.Sequential(nn.Linear(hidden_size, d_ff), activation, nn.Linear(d_ff, hidden_size))


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
    ):
        """Initialize our transformer, uses bert-base defaults

        :param hidden_size: Size of the transformer inputs and outputs (d_model in the paper)
        :param num_heads: The number of heads for multi-headed attention
        :param dropout: A dropout to apply to each sub-blocks outputs
        :param layer_norm_eps: The noise applied in the layer norm calculation
        :param activation: The activation function to use
        :param feed_forward_size:  The optional size of the FFN internal representation.  Defaults to 4*hidden_size
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_ff = feed_forward_size
        self.self_attention = MultiHeadedAttention(hidden_size, num_heads)
        self.self_attention_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.ffn = create_feed_forward_layer(hidden_size, feed_forward_size, activation)
        self.output_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

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


class PreLayerNormTransformerEncoderLayer(nn.Module):
    """A single (pre-layer-norm style) Transformer layer

    This layer implements a pre-layer-norm style Transformer.  Normalization is applied first, and then
    MultiHeadedAttention is applied with optional dropout and added to its input.

    Normalization is again applied and then the FFN block is applied.  Then an MLP layer with a larger size
    is applied followed by an activation and down-projection back to the input size. Dropout is again applied,
    and again we add the output to the input of FFN.

    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
    ):
        """Initialize our transformer, uses bert-base defaults

        :param hidden_size: Size of the transformer inputs and outputs (d_model in the paper)
        :param num_heads: The number of heads for multi-headed attention
        :param dropout: A dropout to apply to each sub-blocks outputs
        :param layer_norm_eps: The noise applied in the layer norm calculation
        :param activation: The activation function to use
        :param feed_forward_size:  The optional size of the FFN internal representation.  Defaults to 4*hidden_size
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_ff = feed_forward_size
        self.self_attention = MultiHeadedAttention(hidden_size, num_heads)
        self.self_attention_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.ffn = create_feed_forward_layer(hidden_size, feed_forward_size, activation)
        self.output_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

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

        residual = x
        y = self.self_attention_layer_norm(x)
        y = residual + self.maybe_dropout(self.self_attention(y, mask))
        residual = y
        y = self.output_layer_norm(y)
        y = residual + self.maybe_dropout(self.ffn(y))
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
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.embeddings_layer_norm = (
            nn.LayerNorm(hidden_size, layer_norm_eps) if do_embeddings_layer_norm else nn.Identity()
        )
        self.embeddings = EmbeddingClass(vocab_size, hidden_size, padding_idx=padding_idx, max_seq_len=max_seq_len)
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size)
                for _ in range(num_layers)
            ]
        )

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
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()


class PreLayerNormTransformerEncoder(nn.Module):
    """A Pre-Layer Norm Transformer Encoder (with no task heads)

    This encoder encapsulates the entire front-end of the Transformer from one-hots up to the final
    encoding.  For tasks like LM and fine-tuning we will inherit this module and provide additional
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
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.embeddings = EmbeddingClass(vocab_size, hidden_size, padding_idx=padding_idx, max_seq_len=max_seq_len)
        self.encoder = nn.ModuleList(
            [
                PreLayerNormTransformerEncoderLayer(
                    hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

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
        for t in self.encoder:
            y = t(y, mask)

        y = self.layer_norm(y)
        return y

    def init_layer_weights(self, module):
        """This not directly used on initialization.  If you want to use it, call `module.apply()` on it

        The base classes do make use of it for MLM and pooling in their constructors
        :param module:
        :return:
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

        # TODO: GPT2 only, move this up into the LM?
        for name, p in module.named_parameters():
            if "ffn.2.weight" in name or "output.weight" in name:
                p.data.normal_(mean=0.0, std=(0.02 / math.sqrt(2 * len(self.encoder))))


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
    ):
        """Initialize our transformer, uses bert-base defaults

        :param hidden_size: Size of the transformer inputs and outputs (d_model in the paper)
        :param num_heads: The number of heads for multi-headed attention
        :param dropout: A dropout to apply to each sub-blocks outputs
        :param layer_norm_eps: The noise applied in the layer norm calculation
        :param activation: The activation function to use
        :param feed_forward_size:  The optional size of the FFN internal representation.  Defaults to 4*hidden_size
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.d_ff = feed_forward_size
        self.self_attention = MultiHeadedAttention(hidden_size, num_heads)
        self.self_attention_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.encoder_attention = MultiHeadedEncoderDecoderAttention(hidden_size, num_heads)
        self.encoder_attention_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.ffn = create_feed_forward_layer(hidden_size, feed_forward_size, activation)
        self.output_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

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
        """
        super().__init__()
        self.padding_idx = padding_idx
        self.encoder_embeddings_layer_norm = (
            nn.LayerNorm(hidden_size, layer_norm_eps) if do_embeddings_layer_norm else nn.Identity()
        )
        self.decoder_embeddings_layer_norm = (
            nn.LayerNorm(hidden_size, layer_norm_eps) if do_embeddings_layer_norm else nn.Identity()
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
                TransformerEncoderLayer(hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(hidden_size, num_heads, dropout, layer_norm_eps, activation, feed_forward_size)
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
        :param src_mask: An optional mask to take in for attention
        :return:
        """
        futures_mask = self.causal_mask[:, :, : dst.shape[1], : dst.shape[1]]
        if dst_mask is not None:
            futures_mask = dst_mask & futures_mask.to(dtype=torch.bool)

        src_enc = self.encoder_embeddings(src)
        src_enc = self.encoder_embeddings_layer_norm(src_enc)
        for t in self.encoder:
            src_enc = t(src_enc, src_mask)

        dst_enc = self.decoder_embeddings(dst)
        dst_enc = self.decoder_embeddings_layer_norm(dst_enc)
        for t in self.decoder:
            dst_enc = t(src_enc, dst_enc, src_mask, futures_mask)
        return dst_enc

    def init_layer_weights(self, module):
        """This not directly used on initialization.  If you want to use it, call `module.apply()` on it

        The base classes do make use of it for MLM and pooling in their constructors
        :param module:
        :return:
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()


class TransformerSequenceGenerator(TransformerEncoderDecoder):
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
        )
        self.output_proj = WeightTiedVocabProjection(self.decoder_embeddings.word_embeddings)

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        dst_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dst_enc = super().forward(src, dst, src_mask, dst_mask)
        y = self.output_proj(dst_enc)

        return y
