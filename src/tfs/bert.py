import torch
import torch.nn as nn
import numpy as np
import os
from typing import Optional
import math
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

    def __init__(self, vocab_dim: int, hidden_dim: int = 768, padding_idx: int = 0, max_seq_length: int = 512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_dim, hidden_dim, padding_idx)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_dim)
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
        """For generation, we will need access to the word_embeddings.  Those are transposed to project from a dense

        :return: The word_embeddings weights
        """
        return self.word_embeddings.weight


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
        vocab_size: int,
        padding_idx: int = 0,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        layer_norm_eps=1e-12,
        activation: nn.Module = nn.GELU(),
        feed_forward_size: Optional[int] = None,
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
        self.embeddings_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.embeddings = BertLearnedPositionalEmbedding(vocab_size, hidden_size, padding_idx=padding_idx)
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
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
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
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
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
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            padding_idx,
            hidden_size,
            num_heads,
            num_layers,
            dropout,
            layer_norm_eps,
            activation,
            feed_forward_size,
        )
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)
        self.activation = activation
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.apply(self.init_layer_weights)

    def output_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Affine transformation from final transformer layer dim size to the vocabulary space

        :param x: transformer output
        :return: word predictions
        """
        return (self.embeddings.word_embeddings.weight.unsqueeze(0) @ x.transpose(1, 2)).transpose(
            1, 2
        ) + self.output_bias

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
