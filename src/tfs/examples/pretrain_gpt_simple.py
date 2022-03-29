import logging
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from tfs.gpt import GPT2TransformerLM, GPTTransformerLM, GPTCreator, GPT2Creator
from tfs.train import SingleDeviceLMTrainer
from tokenizers import Tokenizer
import os

logger = logging.getLogger(__file__)

"""Pre-train a GPT2 model in PyTorch (Simple single file version)

This works for a small dataset that fits in memory.  We will use the SimpleTrainer's train_epochs()
function to train this.

"""


def create_single_file_dataset(tokenizer: Tokenizer, fname: str, seq_len) -> Dataset:

    with open(fname) as rf:
        full_text = rf.read()
        tokens = tokenizer.encode(full_text).ids
        num_samples = len(tokens) // seq_len
        trunc = num_samples * seq_len
        if trunc == num_samples:
            tokens.append(tokens[0])
        x_tensors = torch.tensor(tokens[:trunc])
        y_tensors = torch.tensor(tokens[1 : trunc + 1])
    return TensorDataset(x_tensors.view(num_samples, -1), y_tensors.view(num_samples, -1))


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
    parser = argparse.ArgumentParser(description='Pretrain GPT (simple)')
    parser.add_argument("--model_checkpoint_dir", type=str)
    parser.add_argument("--version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--train_file", type=str, required=True, help='File path to use for train file')
    parser.add_argument("--valid_file", type=str, required=True, help='File path to use for valid file')
    parser.add_argument("--hidden_size", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=1, help="Number train workers")
    parser.add_argument("--seq_len", type=int, default=512, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument("--tok_file", type=str, help="The vocab file", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--decay_type", choices=['cosine', 'linear'], help="The type of learning rate decay scheduler")
    parser.add_argument("--alpha_decay", type=float, default=0.0, help="fraction of learning rate by end of training")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate")
    parser.add_argument("--clip", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--weight_decay", type=float, default=1.0e-2, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1, help="Num training epochs")
    parser.add_argument("--restart_from", type=str, help="Option allows you to restart from a previous checkpoint")
    parser.add_argument("--warmup_fract", type=int, default=0.1, help="Fraction of steps spent warming up")
    parser.add_argument("--plateau_fract", type=int, default=0.0, help="Fraction of steps spent holding at max lr")
    parser.add_argument("--saves_per_epoch", type=int, default=10, help="The number of checkpoints to save per epoch")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.model_checkpoint_dir is None:
        args.model_checkpoint_dir = f'clm-{os.getpid()}'
        if not os.path.exists(args.model_checkpoint_dir):
            os.makedirs(args.model_checkpoint_dir)

    if os.path.isdir(args.tok_file):
        args.tok_file = os.path.join(args.tok_file, 'tokenizer.json')
    tokenizer = Tokenizer.from_file(args.tok_file)
    seq_len = 1024 if args.version == 2 else 512

    if args.restart_from:
        global_step = try_get_global_step(args.restart_from)
        Creator = GPT2Creator if args.version == 2 else GPTCreator
        model = Creator.lm_from_pretrained(args.restart_from, **vars(args))
    else:
        global_step = 0
        GPT = GPT2TransformerLM if args.version == 2 else GPTTransformerLM
        model = GPT(tokenizer.get_vocab_size(), **vars(args))

    trainer = SingleDeviceLMTrainer(
        model,
        global_step=global_step,
        **vars(args),
    )
    logger.info(trainer)
    train_dataset = create_single_file_dataset(tokenizer, args.train_file, seq_len)
    valid_dataset = create_single_file_dataset(tokenizer, args.valid_file, seq_len)

    trainer.train_epochs(train_dataset, valid_dataset, os.path.join(args.model_checkpoint_dir, 'ckpt'), args.epochs)


if __name__ == "__main__":
    main()
