

import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    positional_encoding = np.zeros((max_seq_len, d_model))
    position = np.arange(0, max_seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))

    positional_encoding[:, 0::2] = np.sin(position * div_term)
    positional_encoding[:, 1::2] = np.cos(position * div_term)

    return positional_encoding

# Example usage
max_sequence_length = 100
model_dimension = 512

positional_encodings = get_positional_encoding(max_sequence_length, model_dimension)

print("Positional Encodings:")
print(positional_encodings)


