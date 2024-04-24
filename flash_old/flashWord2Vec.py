


from gensim.models import Word2Vec

def train_word2vec(sentences, vector_size=100, window=5, min_count=1):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count)
    return model

# Example usage
bpe_outputs = [
    ['This', 'is', 'the', 'first', 'sen', 'tence.'],
    ['An', 'other', 'sen', 'tence', 'for', 'B', 'P', 'E.'],
    ['Yet', 'an', 'other', 'example', 'sen', 'tence.'],
    ['This', 'is', 'the', 'fourth', 'sen', 'tence.'],
    ['The', 'final', 'sen', 'tence', 'in', 'the', 'example.']
]

model = train_word2vec(bpe_outputs)

print("Word Embeddings:")
for word in model.wv.key_to_index:
    print(f"{word}: {model.wv[word]}")