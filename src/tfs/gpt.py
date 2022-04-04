import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
from tfs.common import PreLayerNormTransformerEncoder, TransformerEncoder, WeightTiedVocabProjection
import logging

logger = logging.getLogger('tfs')


class GPTLearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for GPT

    The embeddings are a combination of 2 inputs, word embeddings, positional embeddings.
    The positional embedding is a learned vector that uses the index offset of the input token.
    The word embeddings is a learned vector that uses the word one-hots to convert to a dense representation.
    Each of these embeddings are added together in the forward
    """

    def __init__(self, vocab_dim: int, hidden_dim: int = 768, padding_idx: int = 0, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = dropout

    def maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout operator in graph only if training

        TODO: this function could also test dropout to make sure its > 0, pruning an unnecessary op
        if training with no dropout

        :param x: The output of the sub-layer
        :return: A (maybe) dropped out version of the input
        """
        return nn.functional.dropout(x, self.dropout) if self.training else x

    def forward(self, x: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Takes a tensor of shape `[B, T]`.  There is a second parameter, but its ignored here

        :param x: A tensor of word one-hots, shape `[B, T]`
        :param _: Ignored, as we dont have this in GPT
        :return: The sum of the positional, word and token type embeddings
        """
        embed = self.word_embeddings(x)
        position = self.position_embeddings(torch.arange(x.shape[-1], dtype=x.dtype).to(x.device)).unsqueeze(0)
        return self.maybe_dropout(embed + position)

    @property
    def max_seq_len(self):
        self.position_embeddings.weight.shape[0]

    @property
    def weight(self):
        """For generation, we will need access to the word_embeddings.  Those are transposed to project from a dense

        :return: The word_embeddings weights
        """
        return self.word_embeddings.weight


class GPTTransformerLM(TransformerEncoder):
    """Our GPT LM predicts tokens from left-to-right, with post-layer-norm encoders

    The GPT model is a post layer norm Transformer with a causal attention mask
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 512,
        **kwargs,
    ):
        super().__init__(
            GPTLearnedPositionalEmbedding,
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
            max_seq_len,
            do_embeddings_layer_norm=False,
        )
        self.activation = activation

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

        self.output_layer = WeightTiedVocabProjection(self.embeddings.word_embeddings)
        self.apply(self.init_layer_weights)


    def create_loss(self):
        return nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the encoder from the parent, followed by penultimate and output projection

        :param x: A one-hot (long) tensor of shape `[B, T]`
        :param mask: An optional mask to take in for attention
        :param token_type: An optional tensor of 0 or 1, shape `[B, T]`
        :return:
        """
        input_mask = self.causal_mask[:, :, : x.shape[1], : x.shape[1]]
        if mask is not None:
            input_mask = mask & input_mask.to(dtype=torch.bool)

        y = super().forward(x, input_mask)
        y = self.output_layer(y)
        return y


class GPT2TransformerLM(PreLayerNormTransformerEncoder):
    """Our GPT2 LM predicts tokens from left-to-right, with pre-layer-norm encoders

    The GPT2 model is identical to the original GPT model except for the layer norm positioning and the scaling
    of the residual connection initializations (and the max_seq_len is now 1024)
    """

    def __init__(
        self,
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
        max_seq_len: int = 1024,
        **kwargs,
    ):
        super().__init__(
            GPTLearnedPositionalEmbedding,
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
            max_seq_len,
        )
        self.activation = activation

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

        self.output_layer = WeightTiedVocabProjection(self.embeddings.word_embeddings)
        self.apply(self.init_layer_weights)

    def create_loss(self):
        return nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the encoder from the parent, followed by penultimate and output projection

        :param x: A one-hot (long) tensor of shape `[B, T]`
        :param mask: An optional mask to take in for attention
        :param token_type: An optional tensor of 0 or 1, shape `[B, T]`
        :return:
        """
        input_mask = self.causal_mask[:, :, : x.shape[1], : x.shape[1]]
        if mask is not None:
            input_mask = mask & input_mask.to(dtype=torch.bool)

        y = super().forward(x, input_mask)
        y = self.output_layer(y)
        return y


class GPT2TransformerPooledEncoder(PreLayerNormTransformerEncoder):
    """Use our Transformer encoder with a pooling head.

    We will use this model for classification
    """

    def __init__(
            self,
            vocab_size: int,
            padding_idx: int = 0,
            hidden_size: int = 768,
            num_heads: int = 12,
            num_layers: int = 12,
            dropout: float = 0.1,
            layer_norm_eps: float = 1e-12,
            activation: nn.Module = nn.GELU(),
            feed_forward_size: Optional[int] = None,
            output: Optional[nn.Module] = None,
            max_seq_len: int = 1024,
            pool_id: Optional[int] = None,
            **kwargs,
    ):
        """Set up initialization for a (post-layer-norm) Transformer with pooling output.  Defaults to bert-base settings

        :param vocab_size: The size of the input vocabulary
        :param padding_idx: The padding index, defaults to 0
        :param hidden_size: The number of hidden units
        :param num_heads: The number of heads for multi-headed attn.  Should divide evenly into hidden_size
        :param num_layers: The number of transformer layers (MHA+FFN) in the architecture
        :param dropout: The value to apply for dropout
        :param layer_norm_eps: The noising term for layer norm
        :param activation: The activation function to use throughout
        :param feed_forward_size: An optional value to set for the FFN MLP output size, defaults to 4*hidden_size
        :param output: An optional projection layer to apply at the end
        :param max_seq_len: The maximum seq len, for GPT2 this should be 1024
        :param pool_id: An optional integer value to use for the pooling token.  If not set, we use mean pooling
        """
        super().__init__(
            GPTLearnedPositionalEmbedding,
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
            max_seq_len,
        )

        self.pooling = self.mean_pool if pool_id is None else self.pool_by_id
        self.pool_id = pool_id

        self.output = output if output else nn.Identity()
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

        self.apply(self.init_layer_weights)

    def forward(
            self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        :param x: A one-hot (long) tensor of shape `[B, T]`
        :param mask: An optional mask to take in for attention
        :param token_type:
        :return:
        """
        input_mask = self.causal_mask[:, :, : x.shape[1], : x.shape[1]]
        if mask is not None:
            input_mask = mask & input_mask.to(dtype=torch.bool)

        y = self.embeddings(x, token_type)
        for t in self.encoder:
            y = t(y, input_mask)

        y = self.pooling(x, y)
        return self.output(y)

    def pool_by_id(self, inputs, embeddings):
        return embeddings[inputs == self.pool_id]

    def mean_pool(self, inputs, embeddings):
        mask = (inputs != 0)
        seq_lengths = mask.sum(1).float()
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, 0.)
        return embeddings.sum(1)/seq_lengths.unsqueeze(-1)


class GPTCreator:
    @classmethod
    def convert_state_dict(cls, tlm, gpt_state_dict):
        tlm_field_names = set(k for k in tlm.state_dict().keys())
        hf_field_names = gpt_state_dict.keys()

        unused_checkpoint_fields = set(hf_field_names)
        remap = {}

        for field_name in hf_field_names:

            if 'mlp.c_fc.weight' in field_name:
                gpt_state_dict[field_name] = gpt_state_dict[field_name].T
            if 'mlp.c_proj.weight' in field_name:
                gpt_state_dict[field_name] = gpt_state_dict[field_name].T
            new_field_name = field_name.replace('tokens_embed', 'embeddings.word_embeddings')
            new_field_name = new_field_name.replace('positions_embed', 'embeddings.position_embeddings')
            new_field_name = new_field_name.replace('h.', 'encoder.')
            new_field_name = new_field_name.replace('.attn', '.self_attention')
            new_field_name = new_field_name.replace('mlp.c_fc', 'ffn.0')
            new_field_name = new_field_name.replace('mlp.c_proj', 'ffn.2')
            new_field_name = new_field_name.replace('.c_attn.weight', '')
            new_field_name = new_field_name.replace('.c_attn.bias', '')
            new_field_name = new_field_name.replace('.c_proj.weight', '')
            new_field_name = new_field_name.replace('.c_proj.bias', '')

            new_field_name = new_field_name.replace('ln_1', 'self_attention_layer_norm')
            new_field_name = new_field_name.replace('ln_2', 'output_layer_norm')
            if 'attn.c_attn.weight' in field_name:
                q, k, v = torch.chunk(gpt_state_dict[field_name].T, 3, dim=0)
                remap[f'{new_field_name}.query.weight'] = q
                tlm_field_names.remove(f'{new_field_name}.query.weight')
                remap[f'{new_field_name}.key.weight'] = k
                tlm_field_names.remove(f'{new_field_name}.key.weight')
                remap[f'{new_field_name}.value.weight'] = v
                tlm_field_names.remove(f'{new_field_name}.value.weight')
                unused_checkpoint_fields.remove(field_name)
            elif 'attn.c_proj.weight' in field_name:
                remap[f'{new_field_name}.output.weight'] = gpt_state_dict[field_name].T
                unused_checkpoint_fields.remove(field_name)
                tlm_field_names.remove(f'{new_field_name}.output.weight')
            elif 'attn.c_attn.bias' in field_name:
                q, k, v = torch.chunk(gpt_state_dict[field_name], 3, dim=0)
                remap[f'{new_field_name}.query.bias'] = q
                tlm_field_names.remove(f'{new_field_name}.query.bias')
                remap[f'{new_field_name}.key.bias'] = k
                tlm_field_names.remove(f'{new_field_name}.key.bias')
                remap[f'{new_field_name}.value.bias'] = v
                tlm_field_names.remove(f'{new_field_name}.value.bias')
                unused_checkpoint_fields.remove(field_name)
            elif 'attn.c_proj.bias' in field_name:
                remap[f'{new_field_name}.output.bias'] = gpt_state_dict[field_name]
                tlm_field_names.remove(f'{new_field_name}.output.bias')
                unused_checkpoint_fields.remove(field_name)
            elif 'attn.bias' in field_name:
                assert (
                    (gpt_state_dict[field_name].squeeze().squeeze() == tlm.causal_mask).all().item()
                )
                unused_checkpoint_fields.remove(field_name)

            if new_field_name in tlm_field_names:
                tlm_field_names.remove(new_field_name)
                unused_checkpoint_fields.remove(field_name)
                remap[new_field_name] = gpt_state_dict[field_name]

        tlm.load_state_dict(remap, strict=False)
        return tlm_field_names, unused_checkpoint_fields

    @classmethod
    def get_vocab_and_hidden_dims(cls, hf_dict: dict) -> tuple:
        try:
            embeddings_weight = hf_dict[[k for k in hf_dict if 'tokens_embed.weight' in k][0]]
        except:
            embeddings_weight = hf_dict[[k for k in hf_dict if 'embeddings.word_embeddings.weight' in k][0]]
        return embeddings_weight.shape

    @classmethod
    def lm_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = GPTCreator.get_vocab_and_hidden_dims(hf_dict)
        tlm = GPTTransformerLM(vocab_size, **kwargs)
        missing, unused = GPTCreator.convert_state_dict(tlm, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return tlm


class GPT2Creator:
    @classmethod
    def convert_state_dict(cls, tlm, gpt_state_dict):
        tlm_field_names = set(k for k in tlm.state_dict().keys())
        hf_field_names = gpt_state_dict.keys()

        unused_checkpoint_fields = set(hf_field_names)
        remap = {}

        for field_name in hf_field_names:

            if 'mlp.c_fc.weight' in field_name:
                gpt_state_dict[field_name] = gpt_state_dict[field_name].T
            if 'mlp.c_proj.weight' in field_name:
                gpt_state_dict[field_name] = gpt_state_dict[field_name].T
            new_field_name = field_name.replace('wte', 'embeddings.word_embeddings')
            new_field_name = new_field_name.replace('wpe', 'embeddings.position_embeddings')
            new_field_name = new_field_name.replace('h.', 'encoder.')
            new_field_name = new_field_name.replace('.attn', '.self_attention')
            new_field_name = new_field_name.replace('mlp.c_fc', 'ffn.0')
            new_field_name = new_field_name.replace('mlp.c_proj', 'ffn.2')
            new_field_name = new_field_name.replace('.c_attn.weight', '')
            new_field_name = new_field_name.replace('.c_attn.bias', '')
            new_field_name = new_field_name.replace('.c_proj.weight', '')
            new_field_name = new_field_name.replace('.c_proj.bias', '')

            new_field_name = new_field_name.replace('ln_1', 'self_attention_layer_norm')
            new_field_name = new_field_name.replace('ln_2', 'output_layer_norm')
            new_field_name = new_field_name.replace('ln_f', 'layer_norm')
            if 'attn.c_attn.weight' in field_name:
                q, k, v = torch.chunk(gpt_state_dict[field_name].T, 3, dim=0)
                remap[f'{new_field_name}.query.weight'] = q
                tlm_field_names.remove(f'{new_field_name}.query.weight')
                remap[f'{new_field_name}.key.weight'] = k
                tlm_field_names.remove(f'{new_field_name}.key.weight')
                remap[f'{new_field_name}.value.weight'] = v
                tlm_field_names.remove(f'{new_field_name}.value.weight')
                unused_checkpoint_fields.remove(field_name)
            elif 'attn.c_proj.weight' in field_name:
                remap[f'{new_field_name}.output.weight'] = gpt_state_dict[field_name].T
                unused_checkpoint_fields.remove(field_name)
                tlm_field_names.remove(f'{new_field_name}.output.weight')
            elif 'attn.c_attn.bias' in field_name:
                q, k, v = torch.chunk(gpt_state_dict[field_name], 3, dim=0)
                remap[f'{new_field_name}.query.bias'] = q
                tlm_field_names.remove(f'{new_field_name}.query.bias')
                remap[f'{new_field_name}.key.bias'] = k
                tlm_field_names.remove(f'{new_field_name}.key.bias')
                remap[f'{new_field_name}.value.bias'] = v
                tlm_field_names.remove(f'{new_field_name}.value.bias')
                unused_checkpoint_fields.remove(field_name)
            elif 'attn.c_proj.bias' in field_name:
                remap[f'{new_field_name}.output.bias'] = gpt_state_dict[field_name]
                tlm_field_names.remove(f'{new_field_name}.output.bias')
                unused_checkpoint_fields.remove(field_name)
            elif 'attn.bias' in field_name:
                assert (
                    (gpt_state_dict[field_name].squeeze().squeeze() == tlm.causal_mask).all().item()
                )  # torch.tril(torch.ones(gpt_state_dict[field_name].shape[-1], gpt_state_dict[field_name].shape[-1]))).all().item()
                unused_checkpoint_fields.remove(field_name)

            if new_field_name in tlm_field_names:
                tlm_field_names.remove(new_field_name)
                unused_checkpoint_fields.remove(field_name)
                remap[new_field_name] = gpt_state_dict[field_name]

        tlm.load_state_dict(remap, strict=False)
        return tlm_field_names, unused_checkpoint_fields

    @classmethod
    def get_vocab_and_hidden_dims(cls, hf_dict: dict) -> tuple:
        try:

            embeddings_weight = hf_dict[[k for k in hf_dict if 'wte.weight' in k][0]]
        except:
            embeddings_weight = hf_dict[[k for k in hf_dict if 'embeddings.word_embeddings.weight' in k][0]]

        return embeddings_weight.shape

    @classmethod
    def lm_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = GPT2Creator.get_vocab_and_hidden_dims(hf_dict)
        tlm = GPT2TransformerLM(vocab_size, **kwargs)
        missing, unused = GPT2Creator.convert_state_dict(tlm, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return tlm

    @classmethod
    def pooled_enc_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, pool_id=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = GPT2Creator.get_vocab_and_hidden_dims(hf_dict)
        enc = GPT2TransformerPooledEncoder(vocab_size, pool_id=pool_id, **kwargs)
        missing, unused = GPT2Creator.convert_state_dict(enc, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return enc
