# Transformers from Scratch!

Minimal PyTorch implementation of common Transformer architectures (starting with BERT).
There is library code that accompanies the tutorials, which explain how to build these architectures from the ground up.

The structure of the library is simple


Because this is written for a tutorial to explain the modeling and training approach, we currently depend on the
HuggingFace tokenizers library to implement subword tokenization.  I selected it because its fast, and widely used.
There are also other good, fast libraries (like BlingFire) that cover multiple subword approaches, but the library
doesnt support them at this time.

## Pretraining

There are 2 example programs at this time showing how to pretrain BERT from scratch (or continue pre-training on pre-trained BERT)

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

Unlike Wikitext-2, the data in Wikipedia doesnt use any tokenization upfront.
There is a regex used in GPT, which is also close to BERTs preprocessing, which we use in this example.

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

