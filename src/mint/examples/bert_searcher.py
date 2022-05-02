import logging
import argparse
import torch
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from mint.bert import BertCreator
from tokenizers import BertWordPieceTokenizer
import faiss
from dataclasses import dataclass
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__file__)

"""An example program where you can search for nearby embeddings from a search index
"""


@dataclass
class SearchEntry:
    embeddings: np.ndarray
    text: str


@dataclass
class Hit:
    """A hit represents a single search result with a score"""

    sim: float
    id: int
    text: str
    vec: Optional[np.ndarray] = None


class SearchIndex:
    def __init__(self, basename):
        """Reload an existing index"""
        index_filename = f'{basename}.index'
        vec_file = f'{basename}.npz'
        self.index = faiss.read_index(index_filename)
        arr = np.load(vec_file)
        self.data = SearchEntry(arr['embeds'], arr['texts'])

    def search(self, embedding: np.ndarray, num_results: int = 5, return_vec=False) -> List[Hit]:
        sim, I = self.index.search(embedding, num_results)
        sim = sim.reshape(-1)
        I = I.reshape(-1)
        hits = []

        for sim, id in zip(sim, I):
            vec = self.data.embeddings[id]
            text = self.data.text[id]
            hit = Hit(id=id, text=text, sim=sim, vec=vec if return_vec else None)
            hits.append(hit)
        return hits


def main():
    parser = argparse.ArgumentParser(description='An interactive search shell with BERT')
    parser.add_argument("--query", type=str, help="Optional query.  If you pass this we wont use the repl")
    parser.add_argument("--k", type=int, default=3, help="How many K for nearest neighbor results?")
    parser.add_argument("--history_file", type=str, default=".embed_history")
    parser.add_argument(
        "--index", help="Index name.  This is used as the base name for an NPZ and a FAISS index file", default="search"
    )
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
    logger.info("Loaded model")
    search_index = SearchIndex(args.index)
    logger.info("Initialized search index")

    def search(query, k):
        with torch.no_grad():
            tokenized_input = tokenizer.encode(query)
            logger.info(tokenized_input.tokens)
            ids = torch.tensor(tokenized_input.ids, device=args.device).unsqueeze(0)
            embedding = embedder(ids).cpu().numpy()
            print(query)
            print('=' * 50)
            hits = search_index.search(embedding, k)
            for hit in hits:
                print(hit)

    if args.query:
        search(args.query, args.k)
        return

    prompt_name = 'Search>> '
    history = FileHistory(args.history_file)
    while True:
        query = prompt(prompt_name, history=history)
        query = query.strip()
        if query == ':quit' or query == 'quit':
            break
        if query.startswith(':k'):
            query = query.split()
            if len(query) == 2:
                args.k = int(query[-1])
                print(f'Setting k={args.k}')
            continue
        search(query, args.k)


if __name__ == "__main__":
    main()
