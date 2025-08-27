import re
from collections import Counter
import random

def load_corpus_from_file(path: str) -> str:
    """
    Load text from a file and return as a single cleaned string.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    # Simple cleanup: keep words and spaces
    text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def preprocess_corpus(tokens, min_freq=10, unk_token="<UNK>"):
    vocab_count = Counter(tokens)
    # Replace rare words with <UNK>
    processed_tokens = [w if vocab_count[w] >= min_freq else unk_token for w in tokens]
    return processed_tokens

def subsample_corpus(corpus, t=1e-5):
    """
    Subsample frequent words from a tokenized corpus.

    corpus: list of tokens
    t: threshold (default 1e-5, can tune)
    """
    # Count frequencies
    token_counts = Counter(corpus)
    total_tokens = len(corpus)

    # Compute word frequencies
    freqs = {w: c/total_tokens for w, c in token_counts.items()}

    # Compute keep probabilities
    keep_probs = {w: min(1.0, (t/freqs[w])**0.5 + (t/freqs[w]))
                  for w in token_counts}

    # Subsample
    subsampled = [w for w in corpus if random.random() < keep_probs[w]]
    return subsampled

def _tokenize(corpus):
    tokens = corpus.split()
    get_vocab_stats(corpus)
    quit()
    vocab = set(tokens)
    print(sorted(vocab))
    print(len(vocab))
    print(len(tokens))
    word_to_idx = {w : i for i, w in enumerate(vocab)}
    idx_to_word = {i : w for w, i in word_to_idx.items()}
    # print(word_to_idx)
    # print(idx_to_word)

if __name__ == "__main__":
    fn = 'advs.txt'
    text = load_corpus_from_file(fn)
    _tokenize(text)
