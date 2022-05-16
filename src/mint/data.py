import torch
from torch.utils.data import IterableDataset, TensorDataset
import random
import logging
import glob
import gzip
import os
import json
from typing import Callable, Optional, List

logger = logging.getLogger("mint")

try:
    import bz2
except:
    logger.warning("Could not import bzip2 decompression lib")


def jsonl_parser(field: str = "x") -> Callable:
    def get_jsonl(line) -> torch.tensor:
        x = json.loads(line)[field]
        return x if x else None

    return get_jsonl


def gpt2_splitter():
    """This is the tokenizer applied to GPT2.  Its not needed now, as Tokenizers provides this logic

    :return: A function to tokenize a string into tokens (prior to subword splitting)
    """
    import regex

    BPE_PATTERN = regex.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    return lambda text: [w.strip() for w in regex.findall(BPE_PATTERN, text)]


class TextFile:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def _open(self):
        if self.filename.endswith(".gz"):
            self.file = gzip.open(self.filename, mode="rt", encoding="utf-8")
        elif self.filename.endswith(".bz2"):
            self.file = bz2.open(self.filename, mode="rt", encoding="utf-8")
        else:
            self.file = open(self.filename, encoding="utf-8")

    def __enter__(self):
        self._open()
        return self.file

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.file.close()


class RawInfiniteDataset(IterableDataset):
    """Infinite dataset on shards with multiple workers and preprocessing on-the-fly

    This approach roughly follows the algorithm from the original BERT paper.
    We split the files up over the worker threads and we shuffle the files.
    Then we use a shuffle buffer of configurable size to further shuffle each
    cooked tensor

    """

    def __init__(
        self,
        pattern: str,
        tokenizer,
        training=True,
        prefix="[CLS]",
        suffix="[SEP]",
        seq_len: int = 512,
        get_data_fn: Optional[Callable] = None,
        shuf_buf_len: int = 100,
    ):
        super().__init__()
        self.get_data_fn = get_data_fn if get_data_fn else lambda x: x if x else None
        self.tokenizer = tokenizer
        self.start_token = self.tokenizer.token_to_id(prefix)
        self.end_token = self.tokenizer.token_to_id(suffix)
        self.pattern = (
            pattern if not os.path.isdir(pattern) else os.path.join(pattern, "*")
        )

        self.samples = 0
        self.rank = 0
        self.world_size = 1
        self.training = training
        self.seq_len = seq_len
        self.shuffle_buffer_len = shuf_buf_len

        if torch.distributed.is_initialized() and self.training:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.training else None

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
                raise Exception(f"No files of pattern {self.pattern} were found!")
        return files, read_file_order, node_worker_id

    def __iter__(self):
        files, read_file_order, _ = self._init_read_order()
        # If we have multiple files per worker, possibly shuffle the file read order
        shuffle_buffer = []
        tokens = []
        while True:
            if self.training:
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
                                    [self.start_token]
                                    + tokens[: self.seq_len - 2]
                                    + [self.end_token]
                                )
                                tokens = tokens[self.seq_len - 2 :]
                                shuffle_buffer.append(tensor)
                                if len(shuffle_buffer) == self.shuffle_buffer_len:
                                    if self.training:
                                        random.shuffle(shuffle_buffer)
                                    # Drain the shuffle buffer
                                    for element in shuffle_buffer:
                                        yield (element,)
                                    shuffle_buffer = []


class InfinitePreprocessedDataset(IterableDataset):
    """Infinite dataset on shards with multiple workers and preprocessing on-the-fly"""

    def __init__(
        self, pattern: str, training=True, get_data_fn=None, shuf_buf_len: int = 100
    ):
        super().__init__()
        self.pattern = pattern
        self.samples = 0
        self.rank = 0
        self.world_size = 1
        self.training = training
        self.shuf_buf_len = shuf_buf_len
        self.get_data_fn = get_data_fn if get_data_fn else lambda x: x if x else None
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()

    def _get_worker_info(self):
        return torch.utils.data.get_worker_info() if self.training else None

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
                raise Exception(
                    f"No files of pattern {self.pattern} were found in {self.directory}!"
                )
        return files, read_file_order, node_worker_id

    def __iter__(self):
        files, read_file_order, _ = self._init_read_order()
        shuffle_buffer = []
        while True:
            if self.training:
                random.shuffle(read_file_order)
            for file_idx in read_file_order:
                file = files[file_idx]
                with open(file) as rf:
                    lines = rf.readlines()
                    for sample in lines:
                        sample = self.get_data(sample.strip())
                        if sample:
                            shuffle_buffer.append(sample)
                            if len(shuffle_buffer) == self.shuffle_buffer_len:
                                if self.training:
                                    random.shuffle(shuffle_buffer)
                                # Drain the shuffle buffer
                                for element in shuffle_buffer:
                                    yield (element,)
                                shuffle_buffer = []


def read_cls_dataset(
    file: str,
    tokenizer,
    pad_index=0,
    get_data_fn: Optional[Callable] = None,
    max_seq_len=512,
    label_list: Optional[List[str]] = None,
) -> TensorDataset:
    def read_space_delim_line(line: str):
        toks = line.split()
        label = toks[0]
        tokens = " ".join(toks[1:])
        return label, tokens

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
            tokens = torch.tensor(tokenizer.encode(example_str).ids)
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
