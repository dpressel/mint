"""Fine-tune BERT as a cross-encoder or dual-encoder (following SentenceBERT) classifier

This program fine-tunes a pre-trained BERT for an unstructured prediction (classification)
task with 2 text inputs.  This typically corresponds to so-called Natural Language Inference
datasets.

The label space should be ternary, with -1 meaning contradiction, 1 for entailment or 0 for neutral.

The loss is a cross-entropy loss

For dual encoding, the network is shared at the lower layers, up until a pooling operation
is performed for each channel, yielding a fixed width vector for each.  The model should predict for each channel's
vector, are they entailment, contradiction or neutral.  This will yield a model that can be used for distance
queries.

Early stopping is performed on the dataset in order to determine the best checkpoint.

If there is a `test_file` provided in the args, we will run an evaluation on our best checkpoint.

"""

from mint.bert import BertCreator
from mint.train import Average
import json
from typing import Optional, List
from mint.data import TextFile, TensorDataset
from tokenizers import BertWordPieceTokenizer
import argparse
import sys
import os
import torch
import logging
from tqdm import tqdm

PAD_VALUE = 0

logger = logging.getLogger(__file__)


class PairedTextTrainer:
    def __init__(self, model, loss_function, device, lr):
        self.model = model
        self.loss_function = loss_function
        self.epoch = 0
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        logger.info(self.optimizer)

    def valid_epoch(self, valid_loader, phase="valid"):
        self.model.eval()
        valid_loss = Average('valid_loss')
        progress = tqdm(enumerate(valid_loader), total=len(valid_loader))
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for i, batch in progress:
                loss, y, y_pred = self.valid_step(batch)
                valid_loss.update(loss.item())
                y_pred = y_pred.argmax(dim=-1).view(-1)
                y = y.view(-1)
                valid_correct += (y == y_pred).sum()
                valid_total += y.shape[0]

                valid_acc = 100.0 * valid_correct / valid_total
                progress.set_description(
                    f"{phase} epoch {self.epoch + 1}, step {i}: loss {valid_loss.avg:.3f}, accuracy {valid_acc:.2f}%"
                )

        return valid_correct / valid_total

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = Average('train_loss')

        warmup_steps = 1
        if self.epoch == 0:
            warmup_steps = int(0.1 * len(train_loader))
            logger.info("Warmup steps %d", warmup_steps)
        progress = tqdm(enumerate(train_loader), total=len(train_loader))
        train_correct = 0
        train_total = 0

        for i, batch in progress:

            lr_factor = min(1.0, (i + 1) / warmup_steps)
            for p in self.optimizer.param_groups:
                p["lr"] = self.lr * lr_factor

            self.optimizer.zero_grad()
            loss, y, y_pred = self.train_step(batch)
            train_loss.update(loss.item())
            loss.backward()
            self.optimizer.step()
            y_pred = y_pred.argmax(dim=-1).view(-1)
            y = y.view(-1)
            train_correct += (y == y_pred).sum()
            train_total += y.shape[0]
            train_acc = 100.0 * (train_correct / train_total)
            progress.set_description(
                f"train epoch {self.epoch + 1}, step {i}: loss {train_loss.avg:.3f}, accuracy {train_acc:.2f}%"
            )


class DualEncoderTrainer(PairedTextTrainer):
    def train_step(self, batch):
        (x1, x2, y) = batch
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x1, x2, self.model.create_pad_mask(x1), self.model.create_pad_mask(x2))
        loss = self.loss_function(y_pred, y)
        return loss, y, y_pred

    def valid_step(self, batch):
        (x1, x2, y) = batch
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x1, x2, self.model.create_pad_mask(x1), self.model.create_pad_mask(x2))
        loss = self.loss_function(y_pred, y)
        return loss, y, y_pred


class CrossEncoderTrainer(PairedTextTrainer):
    def train_step(self, batch):
        (x, tt, y) = batch
        x = x.to(self.device)
        tt = tt.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x, self.model.create_pad_mask(x), token_type=tt)
        loss = self.loss_function(y_pred, y)
        return loss, y, y_pred

    def valid_step(self, batch):
        (x, tt, y) = batch
        x = x.to(self.device)
        tt = tt.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(x, self.model.create_pad_mask(x), token_type=tt)
        loss = self.loss_function(y_pred, y)
        return loss, y, y_pred


def read_jsonl_dual_cls_dataset(
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
        tokens = [
            torch.tensor(tokenizer.encode(obj[cols[1]]).ids)[:max_seq_len],
            torch.tensor(tokenizer.encode(obj[cols[2]]).ids)[:max_seq_len],
        ]
        padded = [torch.full((max_seq_len,), pad_index, dtype=tokens[0].dtype)] * 2
        padded[0][: len(tokens[0])] = tokens[0]
        padded[1][: len(tokens[1])] = tokens[1]
        return padded + [label]

    if os.path.exists(file + ".x1.th") and os.path.exists(file + ".x2.th") and os.path.exists(file + ".y.th"):
        logger.info(
            "Found cached tensor files, reloading.  If you dont want this, delete *.th from %s", os.path.dirname(file)
        )
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


def read_jsonl_cross_cls_dataset(
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
        x1 = torch.tensor(tokenizer.encode(obj[cols[1]]).ids[:max_seq_len])
        x2 = torch.tensor(tokenizer.encode(obj[cols[2]]).ids[1:])
        x1_tt = torch.zeros_like(x1)
        tokens = torch.cat((x1, x2), -1)[:max_seq_len]
        # The output is going to look like [CLS] t1 [SEP] t2 [SEP]
        padded = torch.full((max_seq_len,), pad_index, dtype=tokens[0].dtype)
        tt = torch.ones_like(padded)
        tt[: len(x1_tt)] = x1_tt
        padded[: len(tokens)] = tokens
        return [padded, tt] + [label]

    if os.path.exists(file + ".x.th") and os.path.exists(file + ".x.th") and os.path.exists(file + ".y.th"):
        logger.info(
            "Found cached tensor files, reloading.  If you dont want this, delete *.th from %s", os.path.dirname(file)
        )
        x_tensor = torch.load(file + ".x.th")
        tt_tensor = torch.load(file + ".tt.th")
        y_tensor = torch.load(file + ".y.th")
        return TensorDataset(x_tensor, tt_tensor, y_tensor), label_list

    x_tensor = []
    tt_tensor = []
    y_tensor = []

    with TextFile(file) as rf:
        for line in rf:
            padded_x, tt_x, label = read_line(line.strip())
            x_tensor.append(padded_x)
            tt_tensor.append(tt_x)
            y_tensor.append(label)
        x_tensor = torch.stack(x_tensor)
        tt_tensor = torch.stack(tt_tensor)
        y_tensor = torch.tensor(y_tensor, dtype=torch.long)
    logger.info("Caching tensors for %s in its parent directory", file)
    torch.save(x_tensor, file + ".x.th")
    torch.save(tt_tensor, file + ".tt.th")
    torch.save(y_tensor, file + ".y.th")
    return TensorDataset(x_tensor, tt_tensor, y_tensor), label_list


def trim_to_shortest_len_dual(batch):
    max_x1_len = max((example[0] != PAD_VALUE).sum() for example in batch)
    max_x2_len = max((example[1] != PAD_VALUE).sum() for example in batch)

    y = torch.stack([example[2] for example in batch])
    x1 = torch.stack([example[0][:max_x1_len] for example in batch])
    x2 = torch.stack([example[1][:max_x2_len] for example in batch])
    return x1, x2, y


def trim_to_shortest_len_cross(batch):
    max_x_len = max((example[0] != PAD_VALUE).sum() for example in batch)
    x = torch.stack([example[0][:max_x_len] for example in batch])
    tt = torch.stack([example[1][:max_x_len] for example in batch])
    y = torch.stack([example[2] for example in batch])

    return x, tt, y


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
    parser.add_argument(
        "--encoder_type",
        choices=["cross-encoder", "dual-encoder"],
        default="dual-encoder",
        help="Train a dual-encoder or a cross-encoder",
    )
    parser.add_argument("--vocab_file", type=str, help="The WordPiece model file", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lowercase", action="store_true", help="Vocab is lower case")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--ckpt_base", type=str, default='ckpt')
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--label_names", type=str, nargs="+", default=["contradiction", "neutral", "entailment"])
    parser.add_argument("--col_names", type=str, nargs="+", default=["label", "sentence1", "sentence2"])
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertWordPieceTokenizer(args.vocab_file, lowercase=args.lowercase)

    if args.encoder_type == "dual-encoder":
        trim_batch_fn = trim_to_shortest_len_dual
        read_fn = read_jsonl_dual_cls_dataset
    else:
        logger.info("Using cross-encoder")
        trim_batch_fn = trim_to_shortest_len_cross
        read_fn = read_jsonl_cross_cls_dataset

    train_set, labels = read_fn(
        args.train_file,
        tokenizer,
        pad_index=0,
        max_seq_len=args.max_seq_len,
        cols=args.col_names,
        label_list=args.label_names,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=trim_batch_fn
    )
    logger.info(labels)
    valid_set, labels = read_fn(
        args.valid_file,
        tokenizer,
        pad_index=0,
        max_seq_len=args.max_seq_len,
        label_list=labels,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=trim_batch_fn
    )

    num_classes = len(labels)
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)
    if args.encoder_type == "dual-encoder":
        model = BertCreator.dual_encoder_from_pretrained(args.model, num_classes=num_classes, **vars(args)).to(
            args.device
        )
        trainer = DualEncoderTrainer(model, loss_function, args.device, args.lr)
    else:
        num_classes = len(labels)
        output_layer = torch.nn.Linear(args.hidden_size, num_classes)
        model = BertCreator.pooled_enc_from_pretrained(args.model, output=output_layer, **vars(args)).to(args.device)
        trainer = CrossEncoderTrainer(model, loss_function, args.device, args.lr)

    best_valid_acc = 0.0

    checkpoint_name = args.ckpt_base + '.pth'
    for epoch in range(args.num_epochs):
        trainer.train_epoch(train_loader)

        valid_acc_fract = trainer.valid_epoch(valid_loader)
        # Early stopping check
        if valid_acc_fract > best_valid_acc:
            best_valid_acc = valid_acc_fract
            acc = 100.0 * valid_acc_fract
            logger.info(f"New best validation accuracy {acc:.2f}%")
            torch.save(model.state_dict(), checkpoint_name)
    if not args.test_file:
        logger.info("No test file provided, exiting")
        sys.exit(1)

    test_set, final_labels = read_fn(
        args.test_file,
        tokenizer,
        pad_index=0,
        max_seq_len=args.max_seq_len,
        label_list=labels,
    )
    if len(final_labels) != num_classes:
        raise Exception("The test set adds new classes with no samples in the training or validation")
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, collate_fn=trim_batch_fn
    )

    best_state = torch.load(checkpoint_name)
    model.load_state_dict(best_state)
    eval_fract = trainer.valid_epoch(test_loader, phase='test')
    eval_acc = 100.0 * eval_fract
    print(f"final test accuracy {eval_acc:.2f}%")


if __name__ == "__main__":
    main()
