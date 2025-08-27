import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import pytorch_lightning as pl

from text_utils import load_corpus_from_file
from text_utils import preprocess_corpus
from text_utils import subsample_corpus
from dataset import Word2VecDataset
from word2vec import Word2Vec
from analysis import visualize_embeddings
from analysis import get_word_vector
from analysis import most_similar_words

if __name__ == "__main__":
    fn = 'stud.txt'
    corpus = load_corpus_from_file(fn)
    tokens = corpus.split()
    tokens = subsample_corpus(tokens, t=1e-1)
    tokens = preprocess_corpus(tokens, min_freq = 5)

    dataset = Word2VecDataset(tokens,
                              window_size=5,
                              negative_samples=15)
    dataloader = DataLoader(dataset,
                            batch_size=300,
                            shuffle = True)


    model = Word2Vec(vocab_size = dataset.vocab_size, embed_dim=50)

    trainer = pl.Trainer(max_epochs=50,
                         accelerator="auto",
                         enable_checkpointing=False,
                         logger=False)

    trainer.fit(model, dataloader)

    word = 'holmes'
    sims = most_similar_words(model, word,
                              dataset.word_to_idx,
                              dataset.idx_to_word,
                              top_k=10)
    print(f"Most similar words to '{word}':")
    for w, score in sims:
        print(f"  {w:12s}: {score:.3f}")


    visualize_embeddings(model,
                         dataset.idx_to_word,
                         max_words=dataset.vocab_size,
                         method = 'umap',
                         use_cosine = True)


