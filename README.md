# Transformers from Scratch!

## A Tiny Library for Transformers from the ground up

Minimal PyTorch implementation of common Transformer architectures.  Currently implements
- GPT
- GPT2
- BERT/RoBERTa

This code was written for a (not-yet complete) Colab Tutorial

The first part is up:

- [BERT from scratch](https://colab.research.google.com/drive/175hnhLkJcXH40tGGpO-1kbBrb2IIcIuT?usp=sharing)

The code here is factored out here as a python package for easy use outside of the tutorial.

Because this is written for a tutorial to explain the modeling and training approach, we currently depend on the
HuggingFace tokenizers library to implement subword tokenization.  I selected it because its fast, and widely used.
There are also other good, fast libraries (like BlingFire) that cover multiple subword approaches, but the library
doesnt support them at this time.

## Pretraining

There are example programs at this time showing how to pretrain from scratch (or continue pre-training on pre-trained models)

### In-memory training on a small dataset
There are 2 pretraining examples, one is a toy example good for small datasets like Wikitext-2.
The loader preprocesses the data and slurps the tensors into a TensorDataset. 
It uses the `SimpleTrainer` to train several epochs.  Because the dataset is small and a Map-style dataset, it makes sense to train a whole epoch and then evaluate a whole test dataset.  For large datasets, I would not recommend this approach.

### Out-of-memory training on a large dataset
The second example uses an infinite IterableDataset to read multiple files (shards) and converts them to tensors on the fly.
This program is a more realistic example of language modeling.

### Out-of-memory preprocessed shards on a large dataset

The library also supports fully preprocessed datasets, but there is no example for that usage at this time.

### Wikipedia

To pretrain on English Wikipedia with this program, you'll need an XML wikipedia dump.
This is usually named `enwiki-latest-pages-articles.xml.bz2` and can be found from the [Wikipedia dump site](https://dumps.wikimedia.org/enwiki/latest/).
For example, this should work for downloading:

```
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```
You also need to use this repository:

```
git clone https://github.com/attardi/wikiextractor
git checkout 16186e290d9eb0eb3a3784c6c0635a9ed7e855c3

```
Here is how I ran it for my example:

```
python -m WikiExtractor.py ${INPUT}/enwiki-latest-pages-articles.xml.bz2 \
       -q --json \
       --processes 7 \
       --output ${OUTPUT}/enwiki-extracted \
       --bytes 100M \
       --compress \
       --links \
       --discard_elements gallery,timeline,noinclude \
       --min_text_length 0 \
       --filter_disambig_pages
```
Regarding the command line above, only use `--compress` if you have bzip2 on your system and your Python can

```python
import bz2
```

In each target generated (e.g. AA, AB, AC), we are going to rename with a prefix (e.g. AA):

```
for file in *.bz2; do mv "$file" "AA_$file"; done;
```
We can then copy these to a single directory, or split them however we would like into train and test


Unlike Wikitext-2, the data in Wikipedia doesnt use any tokenization upfront.
There is a regex used in GPT and RoBERTa, (also similar to BERT's) preprocessing, which we use in this example.

Here is how you can train on multiple workers with DistributedDataParallel:

```
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python -m torch.distributed.launch \
        --node=1 \
        --nproc_per_node=8 \
        --node_rank=0 \
        --master_port=$PORT \
        pretrain_bert_wiki.py \
        --vocab_file /data/k8s/hf-models/bert-base-uncased/vocab.txt \
        --lowercase \
        --train_file "/path/to/enwiki-extracted/train/" \
        --valid_file "/path/to/enwiki-extracted/valid/" \
        --num_train_workers 4 \
        --num_valid_workers 1 --batch_size $B --num_steps $N --saves_per_cycle 1 \
        --train_cycle_size 10000 \
        --eval_cycle_size 500 \
        --distributed

```

## Fine-tuning

The [tune_bert_for_cls](src/tfs/examples/tune_bert_for_cls.py) program is a simple example of fine-tuning
our BERT implementation from scratch. 

## Completer REPL

The [bert_completer](src/tfs/examples/bert_completer.py) program allows you to type in masked strings and
see how BERT would complete them.  When it starts, you can pass `--sample` in order to get sampling from the output,
otherwise it uses the most likely values.  You can switch between the 2 modes at runtime using:

```
BERT>> :sample
```
or 
```
BERT>> :max
```
This example uses `prompt_toolkit` which is not a core dependency, but you can install it like this:
```
pip install .[examples]
```


## More Info Soon

