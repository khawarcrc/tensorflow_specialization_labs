# -----------------------------------------------------------
# Text Processing with TensorFlow & Keras
# -----------------------------------------------------------
# In this script, we'll explore how to:
# 1. Build a vocabulary from sentences using TextVectorization.
# 2. Handle unknown words during vectorization.
# 3. Encode sentences into integer sequences.
# 4. Apply pre-padding to sequences using pad_sequences.
# 5. Use ragged tensors for flexible sequence lengths.
# -----------------------------------------------------------

# -----------------------------------------------------------
# Import all relevant libraries
# -----------------------------------------------------------
import tensorflow as tf                          # Core TensorFlow library
from tensorflow.keras.layers import TextVectorization  # For text encoding
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For sequence padding
import numpy as np                               # For array operations (optional, helpful in preprocessing)
import pprint                                    # For pretty-printing vocab/dictionaries (optional)

# -----------------------------------------------------------
# Step 1: Training sentences (our small dataset / corpus)
# -----------------------------------------------------------
# These sentences will be used to "train" the TextVectorization
# layer. It will scan them and build a fixed vocabulary.
training_sentences = [
    "I love my dog",
    "You love my cat",
    "Do you think my dog is amazing",
    "Cats are great companions"
]

# -----------------------------------------------------------
# Step 2: Create the TextVectorization layer
# -----------------------------------------------------------
# output_mode="int" → each word will be mapped to an integer ID.
# By default:
# - Words are lowercased.
# - Punctuation is stripped.
# - Unknown words map to index 1 ([UNK]).
vectorize_layer = TextVectorization(output_mode="int")

# -----------------------------------------------------------
# Step 3: Adapt the vectorizer on the training sentences
# -----------------------------------------------------------
# The .adapt() method scans the data and creates a vocabulary
# of all unique words. This vocabulary is fixed after adaptation.
vectorize_layer.adapt(training_sentences)

# -----------------------------------------------------------
# Step 4: Show the vocabulary
# -----------------------------------------------------------
# get_vocabulary() → returns list of words with indices.
# Index 0 = padding, Index 1 = [UNK] (unknown words).
print("Vocabulary learned from training data:")
print(vectorize_layer.get_vocabulary())
print()

# -----------------------------------------------------------
# Step 5: Test sentences with some new (unseen) words
# -----------------------------------------------------------
# "really", "loves", and "manatee" were never seen in training,
# so they will be mapped to [UNK] = index 1.
test_sentences = [
    "I really love my dog",
    "My dog loves a manatee"
]

# -----------------------------------------------------------
# Step 6: Encode test sentences into integer sequences
# -----------------------------------------------------------
# Each word is mapped to an index from the vocabulary.
# Unknown words → 1.
encoded = vectorize_layer(test_sentences)
print("Encoded test sentences (with [UNK] for unseen words):")
print(encoded.numpy())
print()

# -----------------------------------------------------------
# Step 7: Convert variable-length sequences into list form
# -----------------------------------------------------------
# The encoded sequences may have different lengths.
# We'll convert them into plain numpy arrays for padding.
sequences = [seq.numpy() for seq in encoded]

# -----------------------------------------------------------
# Step 8: Apply pre-padding using pad_sequences
# -----------------------------------------------------------
# pad_sequences ensures all sequences are same length.
# padding="pre" → zeros are added in front.
padded = pad_sequences(sequences, padding="pre")
print("Pre-padded sequences:")
print(padded)
print()

# -----------------------------------------------------------
# Step 9: Alternative → Use Ragged Tensors
# -----------------------------------------------------------
# Instead of auto-padding, we can tell TextVectorization to
# output ragged tensors (different sequence lengths allowed).
vectorize_ragged = TextVectorization(output_mode="int", ragged=True)
vectorize_ragged.adapt(training_sentences)

ragged_sequences = vectorize_ragged(test_sentences)
print("Ragged sequences (no padding, original lengths kept):")
print(ragged_sequences)
print()

# -----------------------------------------------------------
# Step 10: Pad ragged sequences manually
# -----------------------------------------------------------
# We can still use pad_sequences for ragged tensors if needed.
ragged_list = [seq.numpy() for seq in ragged_sequences]
padded_ragged = pad_sequences(ragged_list, padding="pre")
print("Pre-padded ragged sequences:")
print(padded_ragged)
