import torch
import random
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# a word2vec style dataset with negative sampling
class Word2VecDataset(Dataset):
    def __init__(self,
                 tokens,
                 window_size = 5, # commonly 5-10
                 negative_samples = 15, # typically 15-20 for small data and 2-5 for large
                 ):
        self.window_size = window_size
        self.negative_samples = negative_samples

        vocab = set(tokens)
        self.word_to_idx = {w : i for i, w in enumerate(vocab)}
        self.idx_to_word = {i : w for w, i in self.word_to_idx.items()}
        self.vocab_size = len(vocab)

        self.data = []
        indexed = [self.word_to_idx[w] for w in tokens]

        # generate center, context pairs within window
        for i, center in enumerate(indexed):
            for j in range(-window_size, window_size+1):
                if j==0 or i+j<0 or i+j>len(indexed)-1:
                    continue
                context = indexed[i+j]
                self.data.append((center, context))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]

        negatives = []
        while len(negatives) < self.negative_samples:
            neg = random.randint(0, self.vocab_size-1)
            if neg != center and neg != context:
                negatives.append(neg)

        return center, context, torch.tensor(negatives)
