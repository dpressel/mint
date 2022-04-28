from tfs.bert import BertCreator
from tfs.train import Average
import json
from typing import Optional, Callable, List
from tfs.data import TextFile, TensorDataset
from tokenizers import BertWordPieceTokenizer
import argparse
import sys
import os
import torch
import logging
from tqdm import tqdm

PAD_VALUE = 0

logger = logging.getLogger(__file__)

"""Fine-tune BERT as a dual encoder classifier

This program fine-tunes a pre-trained BERT for an unstructured prediction (classification) task with 2 text
inputs.  This typically corresponds to so-called Natural Language Inference datasets.  

The label space should be ternary, with -1 meaning contradiction, 1 for entailment or 0 for neutral.

The loss is a cross-entropy loss, and the network is shared at the lower layers, up until a pooling operation
is performed for each channel, yielding a fixed width vector for each.  The model should predict for each channel's
vector, are they entailment, contradiction or neutral.  This will yield a model that can be used for distance
queries.

Early stopping is performed on the dataset in order to determine the best checkpoint.

If there is a `test_file` provided in the args, we will run an evaluation on our best checkpoint.

"""

def valid_epoch(epoch, loss_function, model, valid_loader, device, phase="valid"):
    model.eval()
    valid_loss = Average('valid_loss')
    progress = tqdm(enumerate(valid_loader), total=len(valid_loader))
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, (x1, x2, y) in progress:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            y_pred = model(x1, x2, model.create_pad_mask(x1), model.create_pad_mask(x2))
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


def train_epoch(lr, epoch, loss_function, model, optimizer, train_loader, device):
    model.train()
    train_loss = Average('train_loss')

    warmup_steps = 1
    if epoch == 0:
        warmup_steps = int(.1 * len(train_loader))
        logger.info("Warmup steps %d", warmup_steps)
    progress = tqdm(enumerate(train_loader), total=len(train_loader))
    train_correct = 0
    train_total = 0

    for i, (x1, x2, y) in progress:

        lr_factor = min(1.0, (i+1) / warmup_steps)
        for p in optimizer.param_groups:
            p["lr"] = lr * lr_factor

        optimizer.zero_grad()
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        y_pred = model(x1, x2, model.create_pad_mask(x1), model.create_pad_mask(x2))
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

def read_jsonl_paired_cls_dataset(
        file: str,
        tokenizer,
        pad_index=0,
        max_seq_len=512,
        cols: Optional[List[str]] = None,
        label_list: Optional[List[str]] = None,
) -> TensorDataset:

    if cols is None:
        cols = ["label", "sentence1", "sentence2"]
    if label_list is None:
        label_list = ["contradiction", "neutral", "entailment"]

    label2index = {} if not label_list else {k: i for i, k in enumerate(label_list)}
    def read_line(l):
        obj = json.loads(l)
        label = label2index[obj[cols[0]]]
        tokens = [torch.tensor(tokenizer.encode(obj[cols[1]]).ids), torch.tensor(tokenizer.encode(obj[cols[2]]).ids)]
        padded = [torch.full((max_seq_len,), pad_index, dtype=tokens[0].dtype)] * 2
        padded[0][: len(tokens[0])] = tokens[0]
        padded[1][: len(tokens[1])] = tokens[1]
        return padded + [label]


    if os.path.exists(file + ".x1.th") and os.path.exists(file + ".x2.th") and os.path.exists(file + ".y.th"):
        logger.info("Found cached tensor files, reloading.  If you dont want this, delete *.th from %s", os.path.dirname(file))
        x1_tensor = torch.load(file + ".x1.th")
        x2_tensor = torch.load(file + ".x2.th")
        y_tensor = torch.load(file + ".y.th")
        return TensorDataset(x1_tensor, x2_tensor, y_tensor), label_list

    x1_tensor = []
    x2_tensor = []
    y_tensor = []
    with TextFile(file) as rf:
        for line in rf:
            padded_x1, padded_x2, label = read_line(line.strip())

            x1_tensor.append(padded_x1)
            x2_tensor.append(padded_x2)
            y_tensor.append(label)
        x1_tensor = torch.stack(x1_tensor)
        x2_tensor = torch.stack(x2_tensor)
        y_tensor = torch.tensor(y_tensor, dtype=torch.long)
    logger.info("Caching tensors for %s in its parent directory", file)
    torch.save(x1_tensor, file + ".x1.th")
    torch.save(x2_tensor, file + ".x2.th")
    torch.save(y_tensor, file + ".y.th")
    return TensorDataset(x1_tensor, x2_tensor, y_tensor), label_list


def trim_to_shortest_len(batch):
    max_x1_len = max((example[0] != PAD_VALUE).sum() for example in batch)
    max_x2_len = max((example[1] != PAD_VALUE).sum() for example in batch)

    y = torch.stack([example[2] for example in batch])
    x1 = torch.stack([example[0][:max_x1_len] for example in batch])
    x2 = torch.stack([example[1][:max_x2_len] for example in batch])
    return x1, x2, y


def main():
    parser = argparse.ArgumentParser(description='fine-tune BERT for classification (dual text input only)')
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--hidden_size", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=1, help="Number train workers")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--vocab_file", type=str, help="The WordPiece model file", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lowercase", action="store_true", help="Vocab is lower case")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--ckpt_base", type=str, default='ckpt-')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--label_names", type=str, nargs="+", default=["contradiction", "neutral", "entailment"])
    parser.add_argument("--col_names", type=str, nargs="+", default=["label", "sentence1", "sentence2"])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertWordPieceTokenizer(args.vocab_file, lowercase=args.lowercase)
    # TODO: read the pad_index in
    train_set, labels = read_jsonl_paired_cls_dataset(args.train_file, tokenizer, pad_index=0, max_seq_len=args.max_seq_len,
                                                      cols=args.col_names, label_list=args.label_names)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=trim_to_shortest_len
    )
    logger.info(labels)
    valid_set, labels = read_jsonl_paired_cls_dataset(
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
    model = BertCreator.dual_encoder_from_pretrained(args.model, num_classes=num_classes, **vars(args)).to(args.device)
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(optimizer)
    checkpoint_name = args.ckpt_base + '.pth'
    best_valid_acc = 0.0

    device = args.device
    for epoch in range(args.num_epochs):
        train_epoch(args.lr, epoch, loss_function, model, optimizer, train_loader, device)

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

    test_set, final_labels = read_jsonl_paired_cls_dataset(
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
