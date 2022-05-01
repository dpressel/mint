"""Create a search index for Transformer embeddings using faiss
"""
import argparse
import os
import torch
import logging
from bert import BertCreator
import faiss
from typing import List
from tokenizers import BertWordPieceTokenizer
from tfs.data import TextFile
import numpy as np
import json


def padded_batch(seqs: List[torch.Tensor]) -> torch.Tensor:
    max_batch_len = max([len(seq) for seq in seqs])
    pad_batch = torch.zeros((len(seqs), max_batch_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        pad_batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return pad_batch


def read_batch(file, file_type, column, has_header, batch_size, tokenizer, max_seq_len):

    col2index = {}
    if file_type == "jsonl":
        read_fn = lambda x: json.loads(x)[column]
    elif file_type == "txt":
        read_fn = lambda x: x
    elif file_type == "tsv" and not has_header:
        column = int(column)
        read_fn = lambda x: x.split('\t')[column]
    else:
        read_fn = lambda x: x.split('\t')[col2index[column]]

    texts = []
    seqs = []
    with TextFile(file) as rf:
        if has_header and file_type != "jsonl":
            header = next(rf)
            header = header.split('\t')
            col2index.update({h: i for i, h in enumerate(header)})

        for line in rf:
            s = read_fn(line.strip())
            texts.append(s)
            seq = tokenizer.encode(s).ids[:max_seq_len]
            seqs.append(seq)
            if len(seqs) == batch_size:
                batch = padded_batch(seqs)
                seqs = []
                batch_texts = texts
                texts = []
                yield batch, batch_texts
        if seqs:
            yield padded_batch(seqs), texts


def main():
    parser = argparse.ArgumentParser(description="Build a search index using BERT (or SentenceBERT)")
    parser.add_argument("--input", help="Input file")
    parser.add_argument("--index", help="Index name.  This will yield an NPZ and a FAISS index file", default="search")
    parser.add_argument("--model", help="A model path or checkpoint", required=True)
    parser.add_argument("--hidden_size", type=int, default=768, help="Model dimension (and embedding dsz)")
    parser.add_argument("--feed_forward_size", type=int, help="FFN dimension")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length for our embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    parser.add_argument("--pool_type", type=str, default="mean")
    parser.add_argument("--vocab_file", type=str, help="The WordPiece model file", required=True)
    parser.add_argument("--lowercase", action="store_true", help="Vocab is lower case")
    parser.add_argument("--file_type", type=str, choices=["tsv", "txt", "jsonl"], default="txt")
    parser.add_argument("--column", type=str, default="0")
    parser.add_argument(
        "--has_header", action="store_true", help="The file has a header line that must be skipped (or used)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertWordPieceTokenizer(args.vocab_file, lowercase=args.lowercase)
    embedder = (
        BertCreator.pooled_enc_from_pretrained(args.model, use_mlp_layer=False, **vars(args)).eval().to(args.device)
    )

    all_embeddings = []
    all_texts = []
    with torch.no_grad():
        for (batch, texts) in read_batch(
            args.input, args.file_type, args.column, args.has_header, args.batch_size, tokenizer, args.max_seq_len
        ):
            batch = batch.to(device=args.device)
            embedded_batch = embedder(batch, embedder.create_pad_mask(batch)).cpu().numpy()
            all_embeddings.append(embedded_batch)
            all_texts += texts
    all_embeddings = np.vstack(all_embeddings)
    np.savez(args.index + ".npz", embeds=all_embeddings, texts=all_texts)
    index = faiss.IndexFlatL2(args.hidden_size)
    index.add(all_embeddings)
    faiss.write_index(index, args.index + ".index")


if __name__ == '__main__':
    main()
