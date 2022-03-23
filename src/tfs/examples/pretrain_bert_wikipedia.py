import logging
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset
from tfs.bert import BertCreator, NoisingCollator, TransformerMLM
from tfs.train import SimpleLMTrainer
from tfs.data import RawInfiniteDataset, wikipedia_parser, gpt2_splitter
from tokenizers import BertWordPieceTokenizer

import os

logger = logging.getLogger(__file__)

"""Pre-train a BERT/RoBERTa model in PyTorch on all of wikipedia via https://github.com/attardi/wikiextractor

We will process the data on-the-fly from the readers, and shard over multiple workers, and we will
train with the SimpleTrainer's train_steps() function

"""


def create_sharded_dataset(
    tokenizer: BertWordPieceTokenizer,
    glob_path: str,
    is_train,
    seq_len: int = 512,
    start_token='[CLS]',
    end_token='[SEP]',
) -> Dataset:
    dataset = RawInfiniteDataset(
        glob_path,
        tokenizer,
        is_train,
        is_train,
        prefix=start_token,
        suffix=end_token,
        seq_len=seq_len,
        get_data_fn=wikipedia_parser(gpt2_splitter()),
    )
    return dataset


def try_get_global_step(checkpoint_name) -> int:
    """If its a checkpoint we saved the suffix will be -step-{global_step}.pth

    We will assume that any checkpoint we reload has the exact same parameterization as this
    run.  If thats not the case the learning params will be different

    :param checkpoint_name: Either a huggingface pretrained checkpoint or one we saved here
    :return: Int representing the global step
    """
    import re

    match = re.match('(\\S+)-step-(\\d+).pth', checkpoint_name)
    global_step = 0
    if match:
        global_step = int(match[2])
    return global_step


def main():
    parser = argparse.ArgumentParser(description='Pretrain BERT (simple)')
    parser.add_argument("--model_checkpoint_dir", type=str)
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--valid_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--hidden_size", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=1, help="Number train workers")
    parser.add_argument("--nctx", type=int, default=512, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--vocab_file", type=str, help="The WordPiece model file", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument(
        "--decay_type", choices=['cosine', 'linear'], default='cosine', help="The type of learning rate decay scheduler"
    )
    parser.add_argument("--alpha_decay", type=float, default=0.0, help="fraction of learning rate by end of training")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--num_steps", type=int, default=250_000, help="Num training steps")
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="#iters before we update (grad_accum*batch_size=eff_batch_size)"
    )
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_fract", type=int, default=0.1, help="Fraction of steps spent warming up")
    parser.add_argument("--plateau_fract", type=int, default=0.0, help="Fraction of steps spent holding at max lr")
    parser.add_argument("--saves_per_cycle", type=int, default=1, help="The number of checkpoints to save per epoch")
    parser.add_argument("--train_cycle_size", type=int, default=1000, help="The many training steps to run before eval")
    parser.add_argument("--eval_cycle_size", type=int, default=200, help="How many steps to evaluate each time")
    parser.add_argument("--lowercase", action="store_true", help="Vocab is lower case")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.model_checkpoint_dir is None:
        args.model_checkpoint_dir = f'mlm-{os.getpid()}'
        if not os.path.exists(args.model_checkpoint_dir):
            os.makedirs(args.model_checkpoint_dir)

    tokenizer = BertWordPieceTokenizer(args.vocab_file, lowercase=args.lowercase)
    vocab_size = tokenizer.get_vocab_size()
    pad_value = tokenizer.token_to_id('[PAD]')
    mask_value = tokenizer.token_to_id('[MASK]')

    if args.restart_from:
        global_step = try_get_global_step(args.restart_from)
        model = BertCreator.mlm_from_pretrained(args.restart_from, **vars(args))
    else:
        global_step = 0
        model = TransformerMLM(tokenizer.get_vocab_size(), **vars(args))

    trainer = SimpleLMTrainer(
        model,
        global_step=global_step,
        collate_function=NoisingCollator(vocab_size, mask_value, pad_value),
        **vars(args),
    )
    logger.info(trainer)
    trainer.show_lr_plan(args.num_steps)

    train_dataset = create_sharded_dataset(tokenizer, args.train_file, True)
    valid_dataset = create_sharded_dataset(tokenizer, args.valid_file, False)

    trainer.train_steps(
        train_dataset,
        valid_dataset,
        os.path.join(args.model_checkpoint_dir, 'ckpt'),
        args.num_steps,
        args.saves_per_cycle,
        args.train_cycle_size,
        args.eval_cycle_size,
    )


if __name__ == "__main__":
    main()
