# --------------------------------------------------------------
# FULL EXAMPLE: Text Preprocessing with TensorFlow
# --------------------------------------------------------------
# Concepts: Vocabulary building, text-to-integer encoding, 
# padding, truncation, ragged tensors, OOV handling, reverse mapping
# --------------------------------------------------------------

# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization    # Layer for converting text to integer sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Utility for manual padding/truncation
import numpy as np
import pprint   # Pretty-print helper (not mandatory, just for nicer output)


# --------------------------------------------------------------
# 1) Training corpus (sentences we "learn vocabulary" from)
# --------------------------------------------------------------
train_sentences = [
    "I love my dog",
    "I love my cat",
    "you love my dog",
    "do you think my dog is amazing?"
]
# Explanation:
# These sentences act as our training text.
# The TextVectorization layer will scan these and build a "vocabulary" (word -> integer mapping).
# Behind the scenes:
#  - TextVectorization lowercases text by default
#  - Splits text on whitespace/punctuation
#  - Builds a frequency-based vocab (most frequent words get lower IDs)


# --------------------------------------------------------------
# 2) Create vectorization layer
# --------------------------------------------------------------
# output_mode="int" → maps each word to an integer ID
# ragged=False (default) → produces dense, padded tensors
# no output_sequence_length → padding will happen dynamically (to longest sentence in batch)
vectorizer = TextVectorization(output_mode="int")

# Adapt = "fit" the layer on training sentences to learn vocab
vectorizer.adapt(train_sentences)

# Extract the vocabulary
vocab = vectorizer.get_vocabulary()  # Returns list: index → token
print("Vocabulary (index -> token):")
for i, token in enumerate(vocab):
    print(i, repr(token))
print()
# Behind the scenes:
# Special tokens are always included:
# index 0 → "[PAD]" (padding)
# index 1 → "[UNK]" (out of vocabulary token)
# Rest of the words sorted by frequency, then alphabetically if tie.


# --------------------------------------------------------------
# 3) Encode a single sentence
# --------------------------------------------------------------
single = ["I love my dog"]   # Single string inside a list
encoded_single = vectorizer(single)
print("Encoded single sentence (post-padded to batch max):", encoded_single.numpy())
print()
# Explanation:
# Passing the sentence through the vectorizer maps each word to its integer ID.
# If there were multiple sentences in the batch, shorter ones would be post-padded with zeros.


# --------------------------------------------------------------
# 4) Apply vectorizer on a tf.data.Dataset
# --------------------------------------------------------------
dataset = tf.data.Dataset.from_tensor_slices(np.array(train_sentences))
# tf.data.Dataset is TensorFlow's pipeline API for efficient data feeding.

# Map the vectorizer to each element (applies transformation lazily)
mapped = dataset.map(vectorizer)

print("Dataset elements after mapping (first 4 items):")
for i, el in enumerate(mapped.take(4)):
    print(i, el.numpy())   # Each element is now a sequence of token IDs
print()
# Behind the scenes:
# - Dataset stores data efficiently (streams data instead of keeping all in memory).
# - Each element (string) is transformed → integer tensor by vectorizer.


# --------------------------------------------------------------
# 5) Pre-padding with pad_sequences
# --------------------------------------------------------------
# Gather all sequences into a Python list
sequences = [el.numpy() for el in mapped]

# pad_sequences converts list of variable-length sequences into uniform length
# padding="pre" means add zeros at the start
pre_padded = pad_sequences(sequences, padding="pre")
print("Pre-padded sequences (pad_sequences with padding='pre'):")
print(pre_padded)
print()
# Explanation:
# If one sentence is shorter than the max length, zeros are added at the beginning.
# This ensures all sequences have equal length (important for matrix operations in neural nets).


# --------------------------------------------------------------
# 6) Post-padding
# --------------------------------------------------------------
# padding="post" means zeros go at the END of sequences
post_padded = pad_sequences(sequences, padding="post")
print("Post-padded sequences (padding='post'):")
print(post_padded)
print()
# Note:
# This is the default style used by the TextVectorization layer internally.


# --------------------------------------------------------------
# 7) Truncation Example
# --------------------------------------------------------------
# Set maxlen=5 (keep only 5 tokens max)
# truncating="pre" (default) → cut off words from the beginning if too long
truncated = pad_sequences(sequences, padding="pre", maxlen=5, truncating="pre")
print("Truncated to maxlen=5 (default truncating='pre'):")
print(truncated)
print()
# Explanation:
# Longer sentences are chopped down to 5 tokens.
# Useful for very long sentences when you don’t want infinite sequence lengths.


# --------------------------------------------------------------
# 8) Ragged Mode (variable-length tensors)
# --------------------------------------------------------------
# ragged=True means vectorizer outputs tf.RaggedTensor instead of dense padded tensor
vectorizer_ragged = TextVectorization(output_mode="int", ragged=True)
vectorizer_ragged.adapt(train_sentences)

ragged_out = vectorizer_ragged(train_sentences)
print("Ragged output (no padding, original lengths kept):")
print(ragged_out)
print("Converted ragged to python lists:")
print([row.numpy() for row in ragged_out])
print()
# Explanation:
# RaggedTensor allows each row to have different lengths.
# This is more memory efficient than padding everything to the max length.
# But many models expect uniform shapes, so padding is often applied later.


# --------------------------------------------------------------
# 9) Handling Unknown Words (OOV)
# --------------------------------------------------------------
test_sentences = ["I really love my dog", "My dog loves a manatee"]
encoded_test = vectorizer(test_sentences)
print("Encoded test sentences (unknowns -> [UNK]):")
print(encoded_test.numpy())
print()
# Explanation:
# Words like "really", "loves", and "manatee" weren’t in the training vocab.
# They get mapped to the special [UNK] token (index 1).
# This ensures consistent handling of unseen words during inference.


# --------------------------------------------------------------
# 10) Reverse Mapping (decode IDs back to words)
# --------------------------------------------------------------
id_to_token = vectorizer.get_vocabulary()  # index → word mapping

def decode_sequence(seq):
    """
    Convert a sequence of token IDs back to words.
    Skips padding (0).
    """
    return " ".join(
        id_to_token[id] if id < len(id_to_token) else "[OUT-OF-RANGE]"
        for id in seq if id != 0
    )

print("Decode example (first test sentence):")
print(decode_sequence(encoded_test.numpy()[0]))
print()
# Behind the scenes:
# Neural networks only see numbers, not words.
# Decoding back helps us "interpret" predictions and debug models.
