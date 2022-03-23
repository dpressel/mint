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

To pretrain on Wikipedia with this program, you'll need an XML wikipedia dump. This is usally named `enwiki-latest-pages-articles.xml.bz2` and can be found from the Wikipedia dump site.

You also need to use this repository: https://github.com/attardi/wikiextractor
to extract the dumps.  Here is how I ran it for my example:

```
python WikiExtractor.py ${INPUT_PATH}/enwiki-latest-pages-articles.xml.bz2 \
       -q --json \
       --processes 7 \
       --output ${OUTPUT_PATH}/enwiki-extracted \
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

The [tune_bert_for_cls](src/tfs/bert/examples/tune_bert_for_cls.py) program is a simple example of fine-tuning
our BERT implementation using HuggingFace datasets.  This is not a core dependency of the library, but you
can add it using 

```
pip install .[examples]
```


## More Info Soon

