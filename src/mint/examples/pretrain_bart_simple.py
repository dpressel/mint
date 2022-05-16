import logging
import argparse
import torch
from torch.utils.data import Dataset, TensorDataset
from mint.bart import BartCreator, NoisingCollator, BartSequenceGenerator
from mint.train import SingleDeviceSeq2SeqTrainer
from tokenizers import Tokenizer
import os

logger = logging.getLogger(__file__)

"""Pre-train a BART model in PyTorch (Simple single file version)

This works for a small dataset that fits in memory.  We will use the SimpleTrainer's train_epochs()
function to train this.

"""


def create_single_file_dataset(
    tokenizer: Tokenizer, fname: str, seq_len: int = 1024
) -> Dataset:
    bos_token = tokenizer.token_to_id("<s>")
    eos_token = tokenizer.token_to_id("</s>")
    with open(fname) as rf:
        tokens = []
        for line in rf:
            line = line.strip()
            if line:
                line = tokenizer.encode(line, add_special_tokens=False)
                tokens += line.ids

        num_toks = seq_len - 2  # Ignore CLS and SEP
        num_samples = len(tokens) // num_toks * num_toks
        tensors = [
            [bos_token] + tokens[i : i + num_toks] + [eos_token]
            for i in range(0, num_samples, num_toks)
        ]
    tensors = torch.tensor(tensors, dtype=torch.long)
    return TensorDataset(tensors)


def try_get_global_step(checkpoint_name) -> int:
    """If its a checkpoint we saved the suffix will be -step-{global_step}.pth

    We will assume that any checkpoint we reload has the exact same parameterization as this
    run.  If thats not the case the learning params will be different

    :param checkpoint_name: Either a huggingface pretrained checkpoint or one we saved here
    :return: Int representing the global step
    """
    import re

    match = re.match("(\\S+)-step-(\\d+).pth", checkpoint_name)
    global_step = 0
    if match:
        global_step = int(match[2])
    return global_step


def main():
    parser = argparse.ArgumentParser(description="Pretrain BART (simple)")
    parser.add_argument("--model_checkpoint_dir", type=str)
    parser.add_argument(
        "--train_file", type=str, required=True, help="File path to use for train file"
    )
    parser.add_argument(
        "--valid_file", type=str, required=True, help="File path to use for valid file"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=768,
        help="Model dimension (and embedding dsz)",
    )
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument(
        "--num_encoder_layers", type=int, default=6, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=6, help="Number of decoder layers"
    )
    parser.add_argument(
        "--num_train_workers", type=int, default=4, help="Number train workers"
    )
    parser.add_argument(
        "--num_valid_workers", type=int, default=1, help="Number train workers"
    )
    parser.add_argument("--seq_len", type=int, default=1024, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch Size")
    parser.add_argument(
        "--tok_file", type=str, help="The path to the GPT2 tokenizer", required=True
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument(
        "--decay_type",
        choices=["cosine", "linear"],
        help="The type of learning rate decay scheduler",
    )
    parser.add_argument(
        "--alpha_decay",
        type=float,
        default=0.0,
        help="fraction of learning rate by end of training",
    )
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Learning rate")
    parser.add_argument(
        "--clip", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1.0e-2, help="Weight decay"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Num training epochs")
    parser.add_argument(
        "--restart_from",
        type=str,
        help="Option allows you to restart from a previous checkpoint",
    )
    parser.add_argument(
        "--warmup_fract",
        type=float,
        default=0.1,
        help="Fraction of steps spent warming up",
    )
    parser.add_argument(
        "--plateau_fract",
        type=float,
        default=0.0,
        help="Fraction of steps spent holding at max lr",
    )
    parser.add_argument(
        "--saves_per_epoch",
        type=int,
        default=10,
        help="The number of checkpoints to save per epoch",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.model_checkpoint_dir is None:
        args.model_checkpoint_dir = f"s2s-{os.getpid()}"
        if not os.path.exists(args.model_checkpoint_dir):
            os.makedirs(args.model_checkpoint_dir)

    if os.path.isdir(args.tok_file):
        args.tok_file = os.path.join(args.tok_file, "tokenizer.json")

    tokenizer = Tokenizer.from_file(args.tok_file)
    vocab = tokenizer.get_vocab()

    if args.restart_from:
        global_step = try_get_global_step(args.restart_from)

        model = BartCreator.from_pretrained(args.restart_from, **vars(args))
    else:
        global_step = 0
        model = BartSequenceGenerator(tokenizer.get_vocab_size(), **vars(args))
    print(model)
    trainer = SingleDeviceSeq2SeqTrainer(
        model,
        global_step=global_step,
        collate_function=NoisingCollator(vocab),
        **vars(args),
    )
    logger.info(trainer)
    train_dataset = create_single_file_dataset(tokenizer, args.train_file, args.seq_len)
    valid_dataset = create_single_file_dataset(tokenizer, args.valid_file, args.seq_len)

    trainer.train_epochs(
        train_dataset,
        valid_dataset,
        os.path.join(args.model_checkpoint_dir, "ckpt"),
        args.epochs,
    )


if __name__ == "__main__":
    main()
