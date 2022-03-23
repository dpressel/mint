from tfs.bert import BertCreator
from tfs.train import Average
from tokenizers import BertWordPieceTokenizer
from collections import OrderedDict
import argparse
import datasets
import numpy as np
import os
import time
from itertools import chain
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__file__)

"""Fine-tune BERT as a classifier

This program fine-tunes a pre-trained BERT for an unstructured prediction (classification) task using
HuggingFace Datasets.  It works on datasets where the is a single input (no NLI etc.).

"""


def main():
    parser = argparse.ArgumentParser(description='fine-tune BERT for classification (single text input only)')
    parser.add_argument("--model", type=str)
    parser.add_argument('--dataset', help='HuggingFace Datasets id', default=['glue', 'sst2'], nargs='+')
    parser.add_argument("--hidden_size", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_train_workers", type=int, default=4, help="Number train workers")
    parser.add_argument("--num_valid_workers", type=int, default=1, help="Number train workers")
    parser.add_argument("--seq_len", type=int, default=512, help="Max input length")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch Size")
    parser.add_argument("--vocab_file", type=str, help="The WordPiece model file", required=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lowercase", action="store_true", help="Vocab is lower case")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertWordPieceTokenizer(args.vocab_file, lowercase=args.lowercase)
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=args.seq_len)

    def convert_to_features(batch):
        features = {}
        features['y'] = batch['label']
        features['ids'] = [torch.tensor(b.ids) for b in tokenizer.encode_batch(batch['sentence'])]
        return features

    dataset = datasets.load_dataset(*args.dataset)

    train_set = dataset['train']
    train_set = train_set.shuffle().map(convert_to_features, batched=True, batch_size=args.batch_size)
    train_set.set_format(type='torch', columns=['ids', 'y'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)

    valid_set = dataset['validation']
    valid_set = valid_set.map(convert_to_features, batched=True, batch_size=args.batch_size)
    valid_set.set_format(type='torch', columns=['ids', 'y'])
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size)

    num_classes = train_set.features['label'].num_classes
    output_layer = torch.nn.Linear(args.hidden_size, num_classes)
    model = BertCreator.pooled_enc_from_pretrained(args.model, output=output_layer, **vars(args)).to(args.device)
    loss_function = torch.nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info(optimizer)
    checkpoint_name = 'ckpt-' + '-'.join(args.dataset) + '.pth'
    best_valid_acc = 0.0
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = Average('train_loss')
        progress = tqdm(enumerate(train_loader), total=len(train_loader))
        correct = 0
        total = 0
        for i, batch in progress:

            optimizer.zero_grad()
            x = batch['ids'].to(args.device)
            y = batch['y'].to(args.device)
            y_pred = model(x, model.create_pad_mask(x))
            loss = loss_function(y_pred, y)
            train_loss.update(loss.item())
            loss.backward()
            optimizer.step()
            y_pred = y_pred.argmax(dim=-1).view(-1)
            y = y.view(-1)
            correct += (y == y_pred).sum()
            total += y.shape[0]
            acc = 100.0 * (correct / total)
            progress.set_description(f"epoch {epoch+1}, step {i}: loss {train_loss.avg:.3f}, accuracy {acc:.2f}%")

        model.eval()
        valid_loss = Average('valid_loss')
        progress = tqdm(enumerate(valid_loader), total=len(valid_loader))
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch in progress:
                x = batch['ids'].to(args.device)
                y = batch['y'].to(args.device)
                y_pred = model(x, model.create_pad_mask(x))
                loss = loss_function(y_pred, y)
                valid_loss.update(loss.item())
                y_pred = y_pred.argmax(dim=-1).view(-1)
                y = y.view(-1)
                correct += (y == y_pred).sum()
                total += y.shape[0]

                acc = 100.0 * correct / total
                progress.set_description(f"epoch {epoch+1}, step {i}: loss {train_loss.avg:.3f}, accuracy {acc:.2f}%")

            acc_fract = correct / total
            if acc_fract > best_valid_acc:
                best_valid_acc = acc_fract
                acc = 100.0 * acc_fract
                logger.info(f"New best validation accuracy {acc:.2f}%")
                torch.save(model.state_dict(), checkpoint_name)


if __name__ == "__main__":
    main()
