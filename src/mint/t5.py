import torch
import torch.nn as nn
import os
import math
from typing import Optional
from mint.preln import PreLayerNormTransformerSequenceGenerator
import logging
import numpy as np
import re

logger = logging.getLogger('mint')


def _relative_position_bucket(relative_position, is_bidirectional, num_buckets, max_distance):
    """

    Taken from:
    https://github.com/huggingface/transformers/blob/78f346c2b5164695ff4aecc27e2438545f14f9fa/src/transformers/models/t5/modeling_t5.py#L375
    https://github.com/tensorflow/mesh/blob/bbb6ce7917e2a8ef1f3dc6990fcacd4f3b075acd/mesh_tensorflow/transformer/transformer_layers.py#L1014
    """
    relative_buckets = 0
    if is_bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets


class SharedRelativeAttentionBias(nn.Module):
    """T5 relative embedding implementation"""

    def __init__(self, num_heads, is_bidirectional):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = 32
        self.max_distance = 128
        self.is_bidirectional = is_bidirectional
        rel_embedding = torch.nn.init.kaiming_normal_(
            torch.empty((self.num_heads, self.num_buckets), dtype=torch.float), nonlinearity='linear'
        )

        self.relative_attention_bias = nn.Parameter(rel_embedding, requires_grad=True)

    def forward(self, T_k, T_q):
        """Get relative attention bias embedding

        :param T_k: The encoder input length
        :param T_q: The decoder input length
        :return:
        """
        memory_position = torch.arange(T_k).view(1, -1)
        query_position = torch.arange(T_q).view(-1, 1)
        relative_position = memory_position - query_position

        rp_bucket = _relative_position_bucket(
            relative_position, self.is_bidirectional, self.num_buckets, self.max_distance
        )
        relative_attention_bias = self.relative_attention_bias[:, rp_bucket]
        return relative_attention_bias


class MultiHeadedEncoderDecoderRelativeAttentionBias(nn.Module):
    """Multi-headed encoder-decoder attention implementation using scaled dot product

    Converts the input tensors to 3 low-order projections, query, key and value and performs
    multi-headed scaled dot-product attention on them following the Vaswani paper.  The result
    is re-projected a single output representation

    """

    def __init__(self, hidden_size: int, num_heads: int, relative_attention_bias: SharedRelativeAttentionBias):
        """Each block has the same hidden unit size (`d_model` in the paper).  Must be a multiple of num heads

        :param hidden_size: The number of units (both input and output) of the MHA block
        :param num_heads: The number of heads to split into
        """
        super().__init__()

        d_k = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.query.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.key = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.key.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.value = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.value.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.output = nn.Linear(num_heads * d_k, hidden_size, bias=False)
        self.output.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)

        self.num_heads = num_heads
        self.d_k = d_k
        self.relative_attention_bias = relative_attention_bias

    def forward(self, src: torch.Tensor, dst: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-headed encoder-decoder attention with relative bias

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
        dot_prod = query_vec @ key_vec.transpose(-1, -2)
        relative_attention_bias = self.relative_attention_bias(T_k, T_q)
        dot_prod += relative_attention_bias

        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == False, -1e9)

        attn = nn.functional.softmax(dot_prod, dim=-1)
        pre_output = attn @ value_vec

        pre_output = pre_output.transpose(1, 2).contiguous()
        output = self.output(pre_output.view(B, T_q, -1))
        return output


class MultiHeadedRelativeAttentionBias(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, relative_attention_bias: SharedRelativeAttentionBias
    ):  # num_buckets: int = 32, max_distance: int = 128):
        """Each block has the same hidden unit size (`d_model` in the paper).  Must be a multiple of num heads

        :param hidden_size: The number of units (both input and output) of the MHA block
        :param num_heads: The number of heads to split into
        :param is_bidirectional: This will be True for the encoder and false for the decoder

        """
        super().__init__()

        d_k = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.query.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.key = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.key.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.value = nn.Linear(hidden_size, num_heads * d_k, bias=False)
        self.value.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.output = nn.Linear(num_heads * d_k, hidden_size, bias=False)
        self.output.weight.data.normal_(mean=0.0, std=hidden_size ** -0.5)
        self.num_heads = num_heads
        self.d_k = d_k
        self.relative_attention_bias = relative_attention_bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-headed self attention with relative bias

        :param x: A `[B, T, C]` tensor where B is batch, T is time, C is hidden size
        :param mask: An optional mask to apply to the attention matrix
        :return: The attended value vector projected into the output space
        """
        B, T, _ = x.shape
        query_vec = self.query(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        key_vec = self.key(x).view(B, T, self.num_heads, -1).transpose(1, 2)
        value_vec = self.value(x).view(B, T, self.num_heads, -1).transpose(1, 2)

        # [B, H, T_q, D] x [B, H, D, T_k] = [B, H, T_q, T_k]
        dot_prod = query_vec @ key_vec.transpose(-1, -2)
        dot_prod += self.relative_attention_bias(T, T)

        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == False, -1e9)

        attn = nn.functional.softmax(dot_prod, dim=-1)
        pre_output = attn @ value_vec

        pre_output = pre_output.transpose(1, 2).contiguous()
        output = self.output(pre_output.view(B, T, -1))
        return output


def create_feed_forward_layer_no_bias(
    hidden_size: int, feed_forward_size: Optional[int] = None, activation: nn.Module = nn.ReLU()
):
    """Create a feed-forward layer (called FFN in the paper)

    This uses nn.Sequential to string together each part (the MLP and down-projection back to the output size)

    :param hidden_size: The transformer block size (d_model in the paper)
    :param feed_forward_size: The feed-forward layer size, or 4 * hidden_size.
    :param activation: The activation function, defaults to RELU
    :return: An n.Sequential that wraps the whole FFN transformation block
    """
    d_ff = feed_forward_size if feed_forward_size else 4 * hidden_size
    expand = nn.Linear(hidden_size, d_ff, bias=False)
    expand.weight.data.normal_(mean=0.0, std=hidden_size)
    shrink = nn.Linear(d_ff, hidden_size, bias=False)
    shrink.weight.data.normal_(mean=0.0, std=d_ff)
    return nn.Sequential(expand, activation, shrink)


class LayerNormWithoutAdditiveBias(nn.Module):
    """T5 divides by the root-mean-square, with a learned weight and no bias

    The calcuation here is to normalize by the root-mean-square (RMS).
    In CUDA, it may be slightly faster to perform an rsqrt() than doing
    a sqrt() and then dividing:

    http://adrianboeing.blogspot.com/2009/10/timing-square-root-on-gpu.html

    """

    def __init__(self, hidden_dim: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate RMS to normalize, apply learned weight

        :param x:
        :return:
        """
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        y = x * torch.rsqrt(variance + self.eps)
        return self.weight * y


class WordOnlyEmbedding(nn.Module):
    """Embeddings for T5

    The embeddings for T5 are just lookup table embeddings with a bias term provided at each layer.
    No `token_type` is used and the `max_seq_len` is ignored
    """

    def __init__(self, vocab_dim: int, hidden_dim: int = 768, padding_idx: int = 0, max_seq_len: int = 512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx)
        self.word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

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
        enc_shared_relative_attention_bias = SharedRelativeAttentionBias(num_heads, True)
        dec_shared_relative_attention_bias = SharedRelativeAttentionBias(num_heads, False)
        enc_dec_shared_relative_attention_bias = SharedRelativeAttentionBias(num_heads, False)

        class LayerFactory:

            _instance = None

            @staticmethod
            def get_instance():
                """Static access method."""
                if LayerFactory._instance is None:
                    LayerFactory()

                return LayerFactory._instance

            def __init__(self):
                if LayerFactory._instance is not None:
                    raise Exception("Singleton constructor call.  Expected no definition")

                self.encoder_multihead_attention = lambda x, y: MultiHeadedRelativeAttentionBias(
                    x, y, enc_shared_relative_attention_bias
                )
                self.decoder_multihead_attention = lambda x, y: MultiHeadedRelativeAttentionBias(
                    x, y, dec_shared_relative_attention_bias
                )
                self.encoder_decoder_attention = lambda x, y: MultiHeadedEncoderDecoderRelativeAttentionBias(
                    x, y, enc_dec_shared_relative_attention_bias
                )
                self.layer_norm = LayerNormWithoutAdditiveBias
                self.feed_forward = create_feed_forward_layer_no_bias
                LayerFactory._instance = self

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
            LayerFactory.get_instance(),
        )
        # T5 has a pre projection scaling prior to projection
        self.output_proj.pre_scale = hidden_size ** -0.5
        self.enc_shared_relative_attention_bias = enc_shared_relative_attention_bias
        self.dec_shared_relative_attention_bias = dec_shared_relative_attention_bias
        self.enc_dec_shared_relative_attention_bias = enc_dec_shared_relative_attention_bias
        self.apply(self.init_layer_weights)

    def create_loss(self):
        return nn.CrossEntropyLoss(ignore_index=0)

    def init_layer_weights(self, module):
        """This not directly used on initialization.  If you want to use it, call `module.apply()` on it

        The derived classes are expected to call it.  At the moment all of this initialization is set during
        module creation.

        :param module:
        :return:
        """


class T5Creator:
    @classmethod
    def convert_state_dict(cls, tlm, t5_state_dict):
        """Convert the state dict to TFS compatible names

        The relative bias components are stored at layer 0 in HF, and we keep a separate class for each of those
        so we need to be careful WRT to that portion.

        The rest is fairly vanilla, but using a regex replacement makes it much more terse due to naming

        :param tlm:
        :param t5_state_dict:
        :return:
        """
        tlm_field_names = set(k for k in tlm.state_dict().keys())
        hf_field_names = t5_state_dict.keys()

        unused_checkpoint_fields = set(hf_field_names)
        remap = {}

        for field_name in hf_field_names:
            if 'relative_attention_bias.weight' in field_name:
                t5_state_dict[field_name] = t5_state_dict[field_name].T

            new_field_name = field_name.replace('shared.weight', 'encoder_embeddings.word_embeddings.weight')
            new_field_name = re.sub(
                r'(en|de)coder.block.(\d+).layer.0.SelfAttention.k.weight',
                r'\1coder.\2.self_attention.key.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'(en|de)coder.block.(\d+).layer.0.SelfAttention.q.weight',
                r'\1coder.\2.self_attention.query.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'(en|de)coder.block.(\d+).layer.0.SelfAttention.v.weight',
                r'\1coder.\2.self_attention.value.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'(en|de)coder.block.(\d+).layer.0.SelfAttention.o.weight',
                r'\1coder.\2.self_attention.output.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'(en|de)coder.block.(\d+).layer.0.layer_norm.weight',
                r'\1coder.\2.self_attention_layer_norm.weight',
                new_field_name,
            )
            # For encoder block, FFN is 1
            new_field_name = re.sub(
                r'encoder.block.(\d+).layer.1.DenseReluDense.wi.weight', r'encoder.\1.ffn.0.weight', new_field_name
            )
            new_field_name = re.sub(
                r'encoder.block.(\d+).layer.1.DenseReluDense.wo.weight', r'encoder.\1.ffn.2.weight', new_field_name
            )
            new_field_name = re.sub(
                r'encoder.block.(\d+).layer.1.layer_norm.weight', r'encoder.\1.output_layer_norm.weight', new_field_name
            )
            # For decoder block, FFN is 2
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.2.DenseReluDense.wi.weight', r'decoder.\1.ffn.0.weight', new_field_name
            )
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.2.DenseReluDense.wo.weight', r'decoder.\1.ffn.2.weight', new_field_name
            )
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.2.layer_norm.weight', r'decoder.\1.output_layer_norm.weight', new_field_name
            )

            # For decoder block, encoder-decoder attention is 1:
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.1.EncDecAttention.k.weight',
                r'decoder.\1.encoder_attention.key.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.1.EncDecAttention.q.weight',
                r'decoder.\1.encoder_attention.query.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.1.EncDecAttention.v.weight',
                r'decoder.\1.encoder_attention.value.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.1.EncDecAttention.o.weight',
                r'decoder.\1.encoder_attention.output.weight',
                new_field_name,
            )
            new_field_name = re.sub(
                r'decoder.block.(\d+).layer.1.layer_norm.weight',
                r'decoder.\1.encoder_attention_layer_norm.weight',
                new_field_name,
            )
            new_field_name = new_field_name.replace('encoder.final_layer_norm', 'encoder_layer_norm')
            new_field_name = new_field_name.replace('decoder.final_layer_norm', 'decoder_layer_norm')

            new_field_name = new_field_name.replace(
                'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                'enc_shared_relative_attention_bias.relative_attention_bias',
            )
            new_field_name = new_field_name.replace(
                'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                'dec_shared_relative_attention_bias.relative_attention_bias',
            )
            new_field_name = new_field_name.replace(
                'decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight',
                'enc_dec_shared_relative_attention_bias.relative_attention_bias',
            )

            if new_field_name in tlm_field_names:
                tlm_field_names.remove(new_field_name)
                unused_checkpoint_fields.remove(field_name)
                remap[new_field_name] = t5_state_dict[field_name]
        tlm.load_state_dict(remap, strict=False)
        return tlm_field_names, unused_checkpoint_fields

    @classmethod
    def get_vocab_and_hidden_dims(cls, hf_dict: dict) -> tuple:
        try:
            embeddings_weight = hf_dict[[k for k in hf_dict if 'shared.weight' in k][0]]
        except:
            embeddings_weight = hf_dict[[k for k in hf_dict if 'encoder_embeddings.word_embeddings.weight' in k][0]]
        return embeddings_weight.shape

    @classmethod
    def from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = T5Creator.get_vocab_and_hidden_dims(hf_dict)
        kwargs['hidden_size'] = hidden_size
        seq2seq = T5SequenceGenerator(vocab_size, **kwargs)
        missing, unused = T5Creator.convert_state_dict(seq2seq, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return seq2seq


def corrupted_spans(inputs, vocab):
    """N-grams are sampled and replaced by a single variable and labels spans are segments of vars

    :param inputs: Uncorrupted text e.g. `Thank you for inviting me to your party last week .`
    :param vocab: A dictionary of strings to integers
    :return: updated inputs and labels e.g. `Thank you <X> to <Y> week .`|`<pad> <X> for inviting me <Y> your party last <Z>`|
    """
    var_id = max(vocab.values())
    bos_id = 0
    eos_id = vocab.get('</s>')

    span_lengths = np.random.poisson(3, len(inputs))
    masked_indices = np.random.binomial(size=len(inputs), n=1, p=0.15)
    last = 0
    masked = []
    label_values = [bos_id]
    for start in masked_indices.nonzero()[0]:
        if start <= last:
            continue
        span_end = start + span_lengths[start]
        if span_end >= len(inputs) - 1:
            break
        label_values += [var_id] + inputs[start:span_end].tolist()
        masked += inputs[last:start].tolist() + [var_id]
        var_id -= 1
        last = start + span_lengths[start]
    if last < len(inputs):
        masked += inputs[last:].tolist()

    label_values += [eos_id]
    label_values = np.array(label_values)
    return np.array(masked), np.array(label_values)


class NoisingCollator:
    """For each item in a batch, noise it and return noised and denoised tensors

    This function pads on the fly based on the m
    """

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def __call__(self, batch):
        """Take a batch of inputs of X, and convert it to a noised X, Y"""
        noised = []
        denoised = []
        max_x = 0
        max_y = 0
        for x in batch:
            x_noise, x_recon = corrupted_spans(x[0].numpy(), self.vocab)
            if len(x_noise) > max_x:
                max_x = len(x_noise)
            if len(x_recon) > max_y:
                max_y = len(x_recon)

            noised.append(torch.from_numpy(x_noise))
            denoised.append(torch.from_numpy(x_recon))

        x = torch.zeros(
            (
                len(noised),
                max_x,
            ),
            dtype=torch.long,
        )
        y = torch.zeros(
            (
                len(denoised),
                max_y,
            ),
            dtype=torch.long,
        )

        for i in range(len(noised)):
            noised[i] = noised[i][:max_x]
            denoised[i] = denoised[i][:max_y]
            x[i, : len(noised[i])] = noised[i]
            y[i, : len(denoised[i])] = denoised[i]

        return x, y
