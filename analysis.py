import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import torch

import torch
import torch.nn.functional as F

def get_word_vector(model, word, word_to_idx):
    """Return embedding vector for a given word."""
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' not in vocabulary")
    idx = torch.tensor([word_to_idx[word]])
    return model.in_embed(idx).detach()

def most_similar_words(model, word, word_to_idx, idx_to_word, top_k=5):
    """Find top-k most similar words to the given word."""
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' not in vocabulary")

    # Vector for the target word
    word_vec = get_word_vector(model, word, word_to_idx)
    word_vec = F.normalize(word_vec, dim=-1)

    # Normalize all embeddings
    all_embeds = F.normalize(model.in_embed.weight.data, dim=-1)

    # Cosine similarity
    sims = torch.matmul(all_embeds, word_vec.T).squeeze()
    best = torch.topk(sims, top_k + 1)  # +1 because the word itself is included
    results = []
    for idx, score in zip(best.indices, best.values):
        w = idx_to_word[idx.item()]
        # if w != word:  # skip itself
        results.append((w, float(score)))
    return results[:top_k]

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import umap
import torch
import torch.nn.functional as F

def visualize_embeddings(model, idx_to_word, max_words=100, method="tsne", use_cosine=False):
    """
    Visualize embeddings with TSNE, PCA, or UMAP.

    Args:
        model: trained Word2Vec model
        idx_to_word: mapping from index -> word
        max_words: number of words to plot
        method: 'tsne', 'pca', or 'umap'
        use_cosine: if True, use cosine similarity instead of Euclidean
                    (works for TSNE and UMAP)
    """
    # Get embeddings
    embeddings = model.in_embed.weight.data.cpu().numpy()
    vocab_size = embeddings.shape[0]

    # Limit vocabulary for visualization
    n_words = min(max_words, vocab_size)
    words = [idx_to_word[i] for i in range(n_words)]
    X = embeddings[:n_words]

    # Normalize if cosine similarity requested
    if use_cosine:
        X = F.normalize(torch.tensor(X), dim=1).numpy()

    # Dimensionality reduction
    if method == "tsne":
        if use_cosine:
            D = cosine_distances(X)
            reducer = TSNE(n_components=2, metric="precomputed", random_state=42, perplexity=15)
            X_reduced = reducer.fit_transform(D)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=15)
            X_reduced = reducer.fit_transform(X)
    elif method == "pca":
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
    elif method == "umap":
        metric = "cosine" if use_cosine else "euclidean"
        reducer = umap.UMAP(n_components=2, metric=metric, random_state=42)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("method must be 'tsne', 'pca', or 'umap'")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=30)

    for i, word in enumerate(words):
        plt.annotate(word, (X_reduced[i, 0], X_reduced[i, 1]), fontsize=9)

    sim_type = "cosine" if use_cosine else "euclidean"
    plt.title(f"Word2Vec Embeddings ({method.upper()} - {sim_type})")
    plt.show()

