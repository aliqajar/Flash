


import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import sklearn


def get_state(vocab):
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq  
    return pairs


def merge_vocab(pair, vocab):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = vocab[word]
    return v_out


def bpe(input_string, num_merges):
    vocab = {' '.join(word): 1 for word in input_string.split()}
    for i in range(num_merges):
        pairs = get_state(vocab)
        if not pairs:
            break

        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    
    return ' '.join(vocab.keys())



def knn_clustering(encoded_strings, k):
    vectorizer = CountVectorizer(token_pattern=r'\S+')
    X = vectorizer.fit_transform(encoded_strings)

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)

    distances, indices = knn.kneighbors(X)
    clusters = defaultdict(list)

    for i, neightbors in enumerate(indices):
        for neighbor in neightbors:
            clusters[neighbor].append(encoded_strings[i])

    return clusters







