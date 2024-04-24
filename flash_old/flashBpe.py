

import re
from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def bpe(input_string, num_merges):
    vocab = {' '.join(word): 1 for word in input_string.split()}
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return list(vocab.keys())

# Example usage
input_strings = [
    "This is the first sentence.",
    "Another sentence for BPE.",
    "Yet another example sentence.",
    "This is the fourth sentence.",
    "The final sentence in the example."
]

num_merges = 10

bpe_outputs = [bpe(string, num_merges) for string in input_strings]

print("BPE Outputs:")
for output in bpe_outputs:
    print(output)