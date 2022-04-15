from tfs.gpt import GPT2Creator
from tfs.train import Average
from tokenizers import ByteLevelBPETokenizer
from typing import Optional, Callable, List
from torch.utils.data import Dataset, TensorDataset
from tfs.data import TextFile
import argparse
import sys
import torch
import logging
from tqdm import tqdm

PAD_VALUE = 0

logger = logging.getLogger(__file__)

"""Fine-tune GPT2 as a classifier

This program fine-tunes a pre-trained GPT2 for an unstructured prediction (classification) task.
The input is assumed to be a 2-column file with the label first.  The delimiter between columns should
be a space or tab.

Early stopping is performed on the dataset in order to determine the best checkpoint.

It reads the files into a TensorDataset for simplicity, and it trims the batch to the max length
observed in a minibatch.

If there is a `test_file` provided in the args, we will run an evaluation on our best checkpoint.

"""


def read_cls_dataset_gpt2(
    file: str,
    tokenizer,
    pad_index=0,
    get_data_fn: Optional[Callable] = None,
    max_seq_len=1024,
    label_list: Optional[List[str]] = None,
) -> TensorDataset:
    def read_space_delim_line(line: str):
        toks = line.split()
        label = toks[0]
        tokens = ' '.join(toks[1:])
        return label, tokens

    eos = tokenizer.token_to_id("<|endoftext|>")
    if get_data_fn is None:
        get_data_fn = read_space_delim_line

    label2index = {} if not label_list else {k: i for i, k in enumerate(label_list)}
    label_offset = len(label2index)
    x_tensor = []
    y_tensor = []
    with TextFile(file) as rf:
        for line in rf:
            label, example_str = get_data_fn(line.strip())
            if label not in label2index:
                label2index[label] = label_offset
                label_offset += 1
            tokens = torch.tensor(tokenizer.encode(example_str).ids + [eos])
            padded = torch.full((max_seq_len,), pad_index, dtype=tokens.dtype)
            padded[: len(tokens)] = tokens
            x_tensor.append(padded)
            y_tensor.append(label2index[label])
        x_tensor = torch.stack(x_tensor)
        y_tensor = torch.tensor(y_tensor, dtype=torch.long)
    label_list = [0] * label_offset
    for label, idx in label2index.items():
        label_list[idx] = label
    return TensorDataset(x_tensor, y_tensor), label_list


def valid_epoch(epoch, loss_function, model, valid_loader, device, phase="valid"):
    model.eval()
    valid_loss = Average('valid_loss')
    progress = tqdm(enumerate(valid_loader), total=len(valid_loader))
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, (x, y) in progress:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, model.create_pad_mask(x))
            loss = loss_function(y_pred, y)
            valid_loss.update(loss.item())
            y_pred = y_pred.argmax(dim=-1).view(-1)
            y = y.view(-1)
            valid_correct += (y == y_pred).sum()
            valid_total += y.shape[0]

            valid_acc = 100.0 * valid_correct / valid_total
            progress.set_description(
                f"{phase} epoch {epoch + 1}, step {i}: loss {valid_loss.avg:.3f}, accuracy {valid_acc:.2f}%"
            )

    return valid_correct / valid_total


def train_epoch(epoch, loss_function, model, optimizer, train_loader, device):
    model.train()
    train_loss = Average('train_loss')
    progress = tqdm(enumerate(train_loader), total=len(train_loader))
    train_correct = 0
    train_total = 0
    for i, (x, y) in progress:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x, model.create_pad_mask(x))
        loss = loss_function(y_pred, y)
        train_loss.update(loss.item())
        loss.backward()
        optimizer.step()
        y_pred = y_pred.argmax(dim=-1).view(-1)
        y = y.view(-1)
        train_correct += (y == y_pred).sum()
        train_total += y.shape[0]
        train_acc = 100.0 * (train_correct / train_total)
        progress.set_description(
            f"train epoch {epoch + 1}, step {i}: loss {train_loss.avg:.3f}, accuracy {train_acc:.2f}%"
        )


def trim_to_shortest_len(batch):
    max_len = max((example[0] != PAD_VALUE).sum() for example in batch)
    y = torch.stack([example[1] for example in batch])
    x = torch.stack([example[0][:max_len] for example in batch])
    return x, y


def main():
    parser = argparse.ArgumentParser(description='fine-tune GPT for classification (single text input only)')
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--merges_file", type=str, required=True, help="Path to vocab file")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=1, help="Number train workers")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch Size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--ckpt_base", type=str, default='ckpt-')
    parser.add_argument(
        "--pool_type", type=str, choices=["mean", "last"], help="token to use for pooling, defaults to mean pooling"
    )
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merges_file)
    # TODO: read the pad_index in
    train_set, labels = read_cls_dataset_gpt2(args.train_file, tokenizer, pad_index=0, max_seq_len=args.max_seq_len)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=trim_to_shortest_len
    )
    logger.info(labels)
    valid_set, labels = read_cls_dataset_gpt2(
        args.valid_file,
        tokenizer,
        pad_index=0,
        max_seq_len=args.max_seq_len,
        label_list=labels,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=trim_to_shortest_len
    )

    num_classes = len(labels)
    output_layer = torch.nn.Linear(args.hidden_size, num_classes)
    pool_id = tokenizer.token_to_id("<|endoftext|>") if args.pool_type == 'last' else None
    model = GPT2Creator.pooled_enc_from_pretrained(args.model, output=output_layer, pool_id=pool_id, **vars(args)).to(
        args.device
    )
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(optimizer)
    checkpoint_name = args.ckpt_base + '.pth'
    best_valid_acc = 0.0

    device = args.device
    for epoch in range(args.num_epochs):
        train_epoch(epoch, loss_function, model, optimizer, train_loader, device)

        valid_acc_fract = valid_epoch(epoch, loss_function, model, valid_loader, device)
        # Early stopping check
        if valid_acc_fract > best_valid_acc:
            best_valid_acc = valid_acc_fract
            acc = 100.0 * valid_acc_fract
            logger.info(f"New best validation accuracy {acc:.2f}%")
            torch.save(model.state_dict(), checkpoint_name)
    if not args.test_file:
        logger.info("No test file provided, exiting")
        sys.exit(1)

    test_set, final_labels = read_cls_dataset_gpt2(
        args.test_file,
        tokenizer,
        pad_index=0,
        max_seq_len=args.max_seq_len,
        label_list=labels,
    )
    if len(final_labels) != num_classes:
        raise Exception("The test set adds new classes with no samples in the training or validation")
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, collate_fn=trim_to_shortest_len
    )

    best_state = torch.load(checkpoint_name)
    model.load_state_dict(best_state)
    eval_fract = valid_epoch(0, loss_function, model, test_loader, device, phase='test')
    eval_acc = 100.0 * eval_fract
    print(f"final test accuracy {eval_acc:.2f}%")


if __name__ == "__main__":
    main()
