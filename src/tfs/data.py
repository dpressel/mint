import torch
from torch.utils.data import IterableDataset
import random
import logging
import glob
import gzip
import bz2
import json
from typing import Callable

logger = logging.getLogger('tfs')


def jsonl_parser(field: str = 'x') -> Callable:
    def get_jsonl(line) -> torch.tensor:
        x = json.loads(line)[field]
        return x if x else None

    return get_jsonl


def gpt2_splitter():
    import regex

    BPE_PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    return lambda text: [w.strip() for w in regex.findall(BPE_PATTERN, text)]


def wikipedia_parser(splitter=None):
    from bs4 import BeautifulSoup

    tokenizer = splitter if splitter else str.split

    def get_doc(line):
        line = json.loads(line)['text']
        text = BeautifulSoup(line, features="lxml")
        # This is a trivial way to replace, we will just sample other surface terms and use those
        for link in text.find_all('a'):
            surface = link.get_text()
            link.replace_with(surface)
        text = ' '.join(tokenizer(text.get_text()))
        return text

    return get_doc


class TextFile:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def _open(self):
        if self.filename.endswith('.gz'):
            self.file = gzip.open(self.filename, mode='rt', encoding='utf-8')
        elif self.filename.endswith('.bz2'):
            self.file = bz2.open(self.filename, mode='rt', encoding='utf-8')
        else:
            self.file = open(self.filename, encoding="utf-8")

    def __enter__(self):
        self._open()
        return self.file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()


class RawInfiniteDataset(IterableDataset):
    """Infinite dataset on shards with multiple workers and preprocessing on-the-fly"""

    def __init__(
        self,
        pattern: str,
        tokenizer,
        distribute=True,
        shuffle=True,
        prefix='[CLS]',
        suffix='[SEP]',
        seq_len=512,
        get_data_fn=None,
    ):
        super().__init__()
        self.get_data_fn = get_data_fn if get_data_fn else lambda x: x if x else None
        self.tokenizer = tokenizer
        self.start_token = self.tokenizer.token_to_id(prefix)
        self.end_token = self.tokenizer.token_to_id(suffix)
        self.pattern = pattern
        self.samples = 0
        self.rank = 0
        self.world_size = 1
        self.shuffle = shuffle
        self.distribute = distribute
        self.seq_len = seq_len
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.distribute else None

    def _init_read_order(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = self._get_worker_info()
        logger.debug("Globbing %s", self.pattern)
        files = sorted(list(glob.glob(self.pattern)))
        logger.debug("Found %d files", len(files))
        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = self.world_size * num_workers_per_node
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This means the user didnt create more shards than workers
                logger.warning(
                    f"There are no files to read for worker {node_worker_id}, offset {offset}!"
                    + " This might mean that you are passing an incorrect training or validation directory"
                )
            else:
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return files, read_file_order, node_worker_id

    def __iter__(self):
        files, read_file_order, _ = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        tokens = []
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file = files[file_idx]
                with TextFile(file) as rf:
                    for line in rf:
                        line = self.get_data_fn(line.strip())
                        if line:
                            line = self.tokenizer.encode(line, add_special_tokens=False)
                            tokens += line.ids
                            if len(tokens) >= (self.seq_len - 2):
                                tensor = torch.tensor(
                                    [self.start_token] + tokens[: self.seq_len - 2] + [self.end_token]
                                )
                                tokens = tokens[self.seq_len - 2 :]
                                yield (tensor,)


class InfinitePreprocessedDataset(IterableDataset):
    """Infinite dataset on shards with multiple workers and preprocessing on-the-fly"""

    def __init__(self, pattern: str, distribute=True, shuffle=True, get_data_fn=None):
        super().__init__()
        self.pattern = pattern
        self.samples = 0
        self.rank = 0
        self.world_size = 1
        self.shuffle = shuffle
        self.distribute = distribute
        self.get_data_fn = get_data_fn if get_data_fn else lambda x: x if x else None
        if torch.distributed.is_initialized() and distribute:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.distribute else None

    def _init_read_order(self):
        # Each node has the same worker_info, so the unique offsets for each is
        # rank * num_workers + worker_id
        # and the total available workers is world_size * num_workers
        worker_info = self._get_worker_info()
        files = sorted(list(glob.glob(f"{self.directory}/{self.pattern}")))

        if worker_info is None:
            num_workers_per_node = 1
            node_worker_id = 0
        else:
            num_workers_per_node = worker_info.num_workers
            node_worker_id = worker_info.id
        all_workers = self.world_size * num_workers_per_node
        offset = self.rank * num_workers_per_node + node_worker_id
        read_file_order = list(range(offset, len(files), all_workers))
        if not read_file_order:
            if offset > 0:
                # This means the user didnt create more shards than workers
                logger.warning(
                    f"There are no files to read for worker {node_worker_id}, offset {offset}!"
                    + " This might mean that you are passing an incorrect training or validation directory"
                )
            else:
                raise Exception(f"No files of pattern {self.pattern} were found in {self.directory}!")
        return files, read_file_order, node_worker_id

    def __iter__(self):
        files, read_file_order, _ = self._init_read_order()
        while True:
            if self.shuffle:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file = files[file_idx]
                with open(file) as rf:
                    lines = rf.readlines()
                    for sample in lines:
                        sample = self.get_data(sample.strip())
                        if sample:
                            yield (sample,)
