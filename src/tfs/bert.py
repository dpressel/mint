import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
from tfs.common import TransformerEncoder, WeightTiedVocabProjection
import logging

logger = logging.getLogger('tfs')


class BertLearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings for BERT

    The embeddings are a combination of 3 inputs, word embeddings, positional embeddings and
    token_type_embeddings.  Under most circumstances, the token_type_embeddings will be 0.
    The positional embedding is a learned vector that uses the index offset of the input token.
    The word embeddings is a learned vector that uses the word one-hots to convert to a dense representation.
    Each of these embeddings are added together in the forward
    """

    def __init__(self, vocab_dim: int, hidden_dim: int = 768, padding_idx: int = 0, max_seq_len: int = 512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)

    def forward(self, x: torch.Tensor, token_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Takes a tensor of shape `[B, T]` and an optional `token_type` of same shape

        :param x: A tensor of word one-hots, shape `[B, T]`
        :param token_type: An optional tensor of 0 or 1, shape `[B, T]` to identify if first segment or second
        :return: The sum of the positional, word and token type embeddings
        """
        embed = self.word_embeddings(x)

        position = self.position_embeddings(torch.arange(x.shape[-1], dtype=x.dtype).to(x.device)).unsqueeze(0)

        if token_type is None:
            token_type = torch.zeros(1, dtype=x.dtype).to(x.device).unsqueeze(0)

        token_type = self.token_type_embeddings(token_type)
        return embed + position + token_type

    @property
    def weight(self):
        """Access word_embeddings weights

        :return: The word_embeddings weights
        """
        return self.word_embeddings.weight


class TransformerPooledEncoder(TransformerEncoder):
    """Use our Transformer encoder with a pooling head.  For BERT, this head is pretrained on NSP

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
        max_seq_len: int = 512,
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
        """
        super().__init__(
            BertLearnedPositionalEmbedding,
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
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.output = output if output else nn.Identity()
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
        y = self.embeddings(x, token_type)
        y = self.embeddings_layer_norm(y)
        for t in self.encoder:
            y = t(y, mask)
        # TODO: not elegant
        y = torch.tanh(self.pooler(y[:, 0, :]))
        return self.output(y)


class TransformerProjectionEncoder(TransformerEncoder):
    """Use our Transformer encoder with a projection to an output layer.

    We will use this model for tagging
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
        max_seq_len: int = 512,
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
        """
        super().__init__(
            BertLearnedPositionalEmbedding,
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
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.output = output if output else nn.Identity()
        self.apply(self.init_layer_weights)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, token_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """

        :param x: A one-hot (long) tensor of shape `[B, T]`
        :param mask: An optional mask to take in for attention
        :param token_type: An optional tensor of 0 or 1, shape `[B, T]`
        :return:
        """
        y = self.embeddings(x, token_type)
        y = self.embeddings_layer_norm(y)
        for t in self.encoder:
            y = t(y, mask)
        return self.output(y)


class TransformerMLM(TransformerEncoder):
    """Our BERT/RoBERTa transformer for pretraining

    This class just derives the base class and adds the final MLP transformation layer and output projection.
    BERT ties the weights between input embeddings and output embeddings, and adds a bias terms that is initialized
    to 0 and fine-tuned.

    We assume here that we are training RoBERTa style.  IOW, we drop the NSP objective altogether.  This means that
    the pooler from BERT is not going to be trained.

    If you are going to do training on a large amount of data during pretraining, padding and masking the input isnt
    optimal and most people dont do it, instead ensuring that each batch is completely full.  If, for some reason,
    you cannot provide large chunks of contiguous data, training will be less efficient and you will probably
    have a harder time learning the long range deps that Transformers are famous for.  However, we leave that exposed
    here, in the sense that you can pass a mask if you desire.
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
        max_seq_len: int = 512,
        **kwargs,
    ):
        super().__init__(
            BertLearnedPositionalEmbedding,
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
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.activation = activation
        self.output_proj = WeightTiedVocabProjection(self.embeddings.word_embeddings)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self.init_layer_weights)

    def output_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Affine transformation from final transformer layer dim size to the vocabulary space

        :param x: transformer output
        :return: word predictions
        """
        return self.output_proj(x) + self.output_bias

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
        y = super().forward(x)
        y = self.layer_norm(self.activation(self.transform(y)))
        return self.output_layer(y)


class BertCreator:
    @classmethod
    def convert_state_dict(cls, tlm, bert_state_dict):
        tlm_field_names = set(k for k in tlm.state_dict().keys())
        hf_field_names = bert_state_dict.keys()

        unused_checkpoint_fields = set(hf_field_names)
        remap = {}
        for field_name in hf_field_names:
            new_field_name = field_name.replace('bert.encoder', 'encoder')
            new_field_name = new_field_name.replace('bert.embeddings', 'embeddings')
            new_field_name = new_field_name.replace('bert.pooler', 'pooler')
            new_field_name = new_field_name.replace('encoder.layer', 'encoder')
            new_field_name = new_field_name.replace('attention.self', 'self_attention')
            new_field_name = new_field_name.replace('attention.output.LayerNorm', 'self_attention_layer_norm')
            new_field_name = new_field_name.replace('attention.output.dense', 'self_attention.output')
            new_field_name = new_field_name.replace('intermediate.dense', 'ffn.0')
            new_field_name = new_field_name.replace('output.dense', 'ffn.2')
            new_field_name = new_field_name.replace('gamma', 'weight')
            new_field_name = new_field_name.replace('beta', 'bias')
            new_field_name = new_field_name.replace('output.LayerNorm', 'output_layer_norm')
            new_field_name = new_field_name.replace('embeddings.LayerNorm', 'embeddings_layer_norm')
            new_field_name = new_field_name.replace('cls.predictions.bias', 'output_bias')
            new_field_name = new_field_name.replace('cls.predictions.transform.dense', 'transform')
            new_field_name = new_field_name.replace('cls.predictions.transform.LayerNorm', 'layer_norm')
            new_field_name = new_field_name.replace('pooler.dense', 'pooler')
            if new_field_name in tlm_field_names:
                tlm_field_names.remove(new_field_name)
                unused_checkpoint_fields.remove(field_name)
                remap[new_field_name] = bert_state_dict[field_name]

        tlm.load_state_dict(remap, strict=False)
        return tlm_field_names, unused_checkpoint_fields

    @classmethod
    def get_vocab_and_hidden_dims(cls, hf_dict: dict) -> tuple:
        embeddings_weight = hf_dict[[k for k in hf_dict if 'embeddings.word_embeddings.weight' in k][0]]
        return embeddings_weight.shape

    @classmethod
    def mlm_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = BertCreator.get_vocab_and_hidden_dims(hf_dict)
        tlm = TransformerMLM(vocab_size, **kwargs)
        missing, unused = BertCreator.convert_state_dict(tlm, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return tlm

    @classmethod
    def enc_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = BertCreator.get_vocab_and_hidden_dims(hf_dict)
        enc = TransformerEncoder(vocab_size, **kwargs)
        missing, unused = BertCreator.convert_state_dict(enc, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return enc

    @classmethod
    def pooled_enc_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = BertCreator.get_vocab_and_hidden_dims(hf_dict)
        enc = TransformerPooledEncoder(vocab_size, **kwargs)
        missing, unused = BertCreator.convert_state_dict(enc, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return enc

    @classmethod
    def proj_enc_from_pretrained(cls, checkpoint_file_or_dir: str, map_location=None, **kwargs):
        if os.path.isdir(checkpoint_file_or_dir):
            checkpoint = os.path.join(checkpoint_file_or_dir, 'pytorch_model.bin')
        else:
            checkpoint = checkpoint_file_or_dir
        hf_dict = torch.load(checkpoint, map_location=map_location)
        vocab_size, hidden_size = BertCreator.get_vocab_and_hidden_dims(hf_dict)
        enc = TransformerProjectionEncoder(vocab_size, **kwargs)
        missing, unused = BertCreator.convert_state_dict(enc, hf_dict)
        logging.info(f'Unset params: {missing}')
        logging.info(f'Unused checkpoint fields: {unused}')
        return enc


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
