import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
from tfs.common import TransformerEncoderDecoder, TransformerEncoderDecoderLM
import logging

logger = logging.getLogger('tfs')


def create_dst_from_src(input_ids: torch.Tensor, decoder_start_token_id: int = 2):
    dst_ids = torch.ones_like(input_ids)
    dst_ids[:, 0] = decoder_start_token_id
    dst_ids[:, 1:] = input_ids[:, :-1]
    return dst_ids


class BartLearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for BART

    The embeddings are a combination of 2 inputs, word embeddings and positional embeddings
    The word embeddings is a learned vector that uses the word one-hots to convert to a dense representation.
    Each of these embeddings are added together in the forward
    """
    BART_POS_OFFSET = 2

    def __init__(self, vocab_dim: int, hidden_dim: int = 768, padding_idx: int = 0, max_seq_len: int = 1024):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx)
        self.position_embeddings = nn.Embedding(max_seq_len + BartLearnedPositionalEmbedding.BART_POS_OFFSET,
                                                hidden_dim)

    def forward(self, x: torch.Tensor, token_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Takes a tensor of shape `[B, T]` and an optional `token_type` of same shape

        :param x: A tensor of word one-hots, shape `[B, T]`
        :param token_type: Ignored for BART!
        :return: The sum of the positional and word embeddings
        """
        embed = self.word_embeddings(x)

        position = self.position_embeddings(torch.arange(x.shape[-1], dtype=x.dtype).to(
            x.device) + BartLearnedPositionalEmbedding.BART_POS_OFFSET).unsqueeze(0)

        return embed + position

    @property
    def weight(self):
        """Access word_embeddings weights

        :return: The word_embeddings weights
        """
        return self.word_embeddings.weight


class BartEncoderDecoder(TransformerEncoderDecoder):
    def __init__(
            self,
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
    ):
        super().__init__(BartLearnedPositionalEmbedding, vocab_size, padding_idx, hidden_size, num_heads,
                         num_encoder_layers, num_decoder_layers, dropout, layer_norm_eps, activation, feed_forward_size,
                         max_seq_len)


class BartPooledEncoderDecoder(TransformerEncoderDecoder):
    EOS_TOKEN = 2

    def __init__(
            self,
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
            output: Optional[nn.Module] = None,
            max_seq_len: int = 1024,
            **kwargs

    ):
        super().__init__(BartLearnedPositionalEmbedding, vocab_size, padding_idx, hidden_size, num_heads,
                         num_encoder_layers, num_decoder_layers, dropout, layer_norm_eps, activation, feed_forward_size,
                         max_seq_len)

        self.output = output if output else nn.Identity()

    def forward(
            self, src: torch.Tensor, dst: Optional[torch.Tensor] = None, src_mask: Optional[torch.Tensor] = None,
            dst_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        dst = create_dst_from_src(src)
        dst_enc = super().forward(src, dst, src_mask, dst_mask)

        eos_mask = dst.eq(BartPooledEncoderDecoder.EOS_TOKEN)
        eos = dst_enc[eos_mask]
        pooled_output = eos.view(dst_enc.shape[0], -1, dst_enc.shape[-1])[:, -1]
        y = self.output(pooled_output)

        return y


class BartEncoderDecoderLM(TransformerEncoderDecoderLM):
    def __init__(
            self,
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
    ):
        super().__init__(BartLearnedPositionalEmbedding, vocab_size, padding_idx, hidden_size, num_heads,
                         num_encoder_layers, num_decoder_layers, dropout, layer_norm_eps, activation, feed_forward_size,
                         max_seq_len)


class BartCreator:
    @classmethod
    def convert_state_dict(cls, tlm, bert_state_dict):
        """Convert the state dict to TFS compatible names

        The encoder token embeddings (AKA word_embeddings) are shared with the decoder token embeddings, and
        in the HF implementation, this is done via `self.shared` so all 3 items are in the original checkpoint,
        and we only need one of them.  We have tied these together by assignment already, so loading the encoder's
        word embeddings updates the decoder word embeddings too

        Note that the positional embeddings are different for encoder and decoder, so these are not shared and both
        are loaded

        :param tlm:
        :param bert_state_dict:
        :return:
        """
        tlm_field_names = set(k for k in tlm.state_dict().keys())
        hf_field_names = bert_state_dict.keys()

        unused_checkpoint_fields = set(hf_field_names)
        remap = {}
        for field_name in hf_field_names:
            new_field_name = field_name.replace('encoder.embed_tokens', 'encoder_embeddings.word_embeddings')
            new_field_name = new_field_name.replace('encoder.embed_positions', 'encoder_embeddings.position_embeddings')
            new_field_name = new_field_name.replace('decoder.embed_positions', 'decoder_embeddings.position_embeddings')
            new_field_name = new_field_name.replace('encoder.layernorm_embedding', 'encoder_embeddings_layer_norm')
            new_field_name = new_field_name.replace('decoder.layernorm_embedding', 'decoder_embeddings_layer_norm')

            new_field_name = new_field_name.replace('self_attn', 'self_attention')
            new_field_name = new_field_name.replace('encoder_attn', 'encoder_attention')
            new_field_name = new_field_name.replace('k_proj', 'key')
            new_field_name = new_field_name.replace('q_proj', 'query')
            new_field_name = new_field_name.replace('v_proj', 'value')
            new_field_name = new_field_name.replace('out_proj', 'output')
            new_field_name = new_field_name.replace('.layers', '')
            new_field_name = new_field_name.replace('attention.output.dense', 'self_attention.output')
            new_field_name = new_field_name.replace('fc1', 'ffn.0')
            new_field_name = new_field_name.replace('fc2', 'ffn.2')
            new_field_name = new_field_name.replace('final_layer_norm', 'output_layer_norm')
            if new_field_name in tlm_field_names:
                tlm_field_names.remove(new_field_name)
                unused_checkpoint_fields.remove(field_name)
                remap[new_field_name] = bert_state_dict[field_name]

        tlm.load_state_dict(remap, strict=False)
        return tlm_field_names, unused_checkpoint_fields

    @classmethod
    def get_vocab_and_hidden_dims(cls, hf_dict: dict) -> tuple:
        embeddings_weight = hf_dict[[k for k in hf_dict if 'encoder.embed_tokens.weight' in k][0]]
        return embeddings_weight.shape

    @classmethod
    def from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = BartCreator.get_vocab_and_hidden_dims(hf_dict)
        seq2seq = BartEncoderDecoderLM(vocab_size, **kwargs)
        missing, unused = BartCreator.convert_state_dict(seq2seq, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return seq2seq

    @classmethod
    def pooled_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = BartCreator.get_vocab_and_hidden_dims(hf_dict)
        seq2seq = BartPooledEncoderDecoder(vocab_size, **kwargs)
        missing, unused = BartCreator.convert_state_dict(seq2seq, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return seq2seq


def noise_inputs(inputs, vocab_size, mask_value, ignore_prefix=True, ignore_suffix=True, mask_prob=0.15, pad_value=0):
    """Apply the BERT masking algorithm

    Identify `mask_prob` fraction of inputs to be corrupted.  80% of those are [MASK] replaced, 10% are
    corrupted with an alternate word, 10% remain the same.  All of the other values are 0 in the label (output)

    :param inputs: An array of one-hot integers
    :param vocab_size: The size of the vocab
    :param mask_value: The token ID for [MASK]
    :param ignore_prefix: Should we avoid masking the first token (usually yeah)
    :param ignore_suffix: Should we avoid masking the suffix (probably, yeah)
    :param mask_prob: A fraction to corrupt (usually 15%)
    :param pad_value: The value for pad tokens in the inputs
    :return: the corrupted inputs and the truth labels for those inputs, both equal length
    """
    labels = np.copy(inputs)
    masked_indices = np.random.binomial(size=len(inputs), n=1, p=mask_prob)
    # make sure if the input is padded we dont mask
    masked_indices = masked_indices & (labels != pad_value)
    # ensure at least one token is masked
    masked_indices[np.random.randint(1, sum(labels != pad_value))] = 1
    if ignore_prefix:
        masked_indices[0] = 0
    if ignore_suffix:
        masked_indices[-1] = 0
    # Anything not masked is 0 so no loss
    labels[masked_indices == 0] = 0
    # Of the masked items, mask 80% of them with [MASK]
    indices_replaced = np.random.binomial(size=len(inputs), n=1, p=0.8)
    indices_replaced = indices_replaced & masked_indices
    inputs[indices_replaced == 1] = mask_value
    indices_random = np.random.binomial(size=len(inputs), n=1, p=0.5)
    # Replace 10% of them with random words, rest preserved for auto-encoding
    indices_random = indices_random & masked_indices & ~indices_replaced
    # Dont predict [PAD] which is zero for bert and 1 for RoBERTa
    # We will assume here that PAD is one of the tokens near the beginning of the vocab
    random_words = np.random.randint(low=pad_value + 1, high=vocab_size - 1, size=len(inputs))
    inputs[indices_random == 1] = random_words[indices_random == 1]
    return inputs, labels


class NoisingCollator:
    """For each item in a batch, noise it and return noised and denoised tensors"""

    def __init__(self, vocab_size, mask_value, pad_value=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_value = mask_value
        self.pad_value = pad_value

    def __call__(self, batch):
        """Take a batch of inputs of X, and convert it to a noised X, Y"""
        noised = []
        denoised = []
        for x in batch:
            x_noise, x_recon = noise_inputs(x[0].numpy(), self.vocab_size, self.mask_value, self.pad_value)
            noised.append(torch.from_numpy(x_noise))
            denoised.append(torch.from_numpy(x_recon))

        noised = torch.stack(noised)
        denoised = torch.stack(denoised)
        return noised, denoised
