import logging
import argparse
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tfs.gpt import GPTCreator, GPT2Creator
from tokenizers import Tokenizer
import os
from tfs.train import Average
from tqdm import tqdm
import time
import math

logger = logging.getLogger(__file__)


def create_single_file_dataset(tokenizer, fname: str, seq_len: int = 1024) -> Dataset:

    with open(fname) as rf:
        full_text = rf.read()

        num_words = len(full_text.split())
        tokens = tokenizer.encode(full_text).ids
        num_samples = len(tokens) // seq_len
        trunc = num_samples * seq_len
        if trunc == num_samples:
            tokens.append(tokens[0])
        x_tensors = torch.tensor(tokens[:trunc])
        y_tensors = torch.tensor(tokens[1 : trunc + 1])
        num_subwords = y_tensors.nelement()
    return TensorDataset(x_tensors.view(num_samples, -1), y_tensors.view(num_samples, -1)), num_words, num_subwords


def main():
    parser = argparse.ArgumentParser(description='GPT perplexity on test set')
    parser.add_argument("--model", type=str, required=True, help="Start from a model")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--merges_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--file", type=str, help="A test file")
    parser.add_argument("--version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tok_model = os.path.join(args.model if os.path.isdir(args.model) else os.path.dirname(args.model), 'tokenizer.json')
    print(tok_model)
    tokenizer = Tokenizer.from_file(tok_model)
    Creator = GPT2Creator if args.version == 2 else GPTCreator
    seq_len = 1024 if args.version == 2 else 512
    model = Creator.lm_from_pretrained(args.model).eval()
    model.to(args.device)
    loss_function = model.create_loss().to(args.device)
    eval_dataset, num_words, num_subwords = create_single_file_dataset(tokenizer, args.file, seq_len=seq_len)
    logger.info(
        "Num samples in dataset [%d], num words [%d], num subwords [%d]", len(eval_dataset), num_words, num_subwords
    )
    eval_data_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    compute_perplexity(args.device, eval_data_loader, loss_function, model, num_subwords, num_words)


def compute_perplexity(device, eval_data_loader, loss_function, model, num_subwords, num_words=None):
    start = time.time()
    progress = tqdm(enumerate(eval_data_loader), total=len(eval_data_loader))
    avg = Average('avg_loss')
    with torch.no_grad():
        for iters, (x, y) in progress:
            x = x.to(device=device)
            y = y.to(device=device)
            logits = model(x)
            loss = loss_function(logits.reshape(-1, model.vocab_size), y.view(-1))
            avg.update(loss.item())
    loss = avg.avg
    ppl = math.exp(loss)
    elapsed = time.time() - start
    print(f"Evaluation completed [{elapsed: .2f}s]")
    print(f"Subword loss & perplexity: loss {loss:.4f}. perplexity (subword): {ppl: .3f}")

    if num_words is not None:
        word_ppl = math.exp(loss * (num_subwords - 1) / (num_words - 1))
        print(f"Word level perplexity {word_ppl: .3f}")


if __name__ == "__main__":
    main()
