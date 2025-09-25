"""
Lanigan's Ball - Single Song Text Generation Lab
------------------------------------------------

This notebook-style Python script contains a complete, self-contained lab that demonstrates
how to:
  1. take a single song (a short corpus),
  2. build a text-vectorization pipeline using tf.keras.layers.TextVectorization,
  3. create training sequences (n-gram style) from lines of the song,
  4. build and train a small language model (Embedding -> BiLSTM -> Dense softmax),
  5. inspect the vocabulary, token indices, and training data,
  6. generate new text using sampling with temperature and repetition-control,
  7. save/restore vocabulary and model, and
  8. discuss limitations, alternatives, and next steps.

This file intends to be the SINGLE SOURCE OF TRUTH for this lab: all conceptual notes,
shape comments, and implementation rationale are included as inline comments. If you
want to convert this script into a Jupyter notebook, copy each commented block into
cells (the comments will serve as explanations / markdown cells).

NOTE: This script uses TensorFlow 2.x. If you do not have TensorFlow installed, run:
    pip install -U tensorflow

All code is runnable as-is. Replace `RAW_SONG_TEXT` below with your actual single-song
text if you want to experiment with different content.

-- Author: ChatGPT (detailed educational lab)
-- Date: 2025-09-24
"""

# ---------------------------
# 1) Imports & environment
# ---------------------------
import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Set a reproducible seed for numpy / python random / tensorflow when possible.
# Reproducibility is difficult when using GPU nondeterministic ops; this is a best-effort.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# 2) The single-song corpus
# ---------------------------
# This is a small example corpus based on the traditional Irish tune often referenced
# in examples. It's intentionally short so you can see the behaviour when training a
# language model on a tiny dataset. Replace RAW_SONG_TEXT with a larger corpus to
# get better prediction quality.
RAW_SONG_TEXT = """
In the town of Athy one Jeremy Lanigan
Battered away till he had no more pelf
His father died and left him a farm in the valley
And soon he began to prosper himself
He had in his house a good wife and a smart son
And a daughter who was the pride of her town
They danced at the fair and they danced at the wedding
And they danced till the night came down
"""

# Explanation:
# - RAW_SONG_TEXT contains the whole song as one string. Each line is a 'sentence' for our
#   simple corpus (we will split on newline characters). In practice you may want to do
#   more advanced sentence splitting, but for traditional songs, newlines are a good proxy
#   for musical lines.

# Split by newline, trim whitespace, and drop empty lines.
sentences = [line.strip() for line in RAW_SONG_TEXT.split('\n') if line.strip()]
print(f"Number of sentences / lines in corpus: {len(sentences)}")

# Print the lines so we can visually confirm the corpus.
for i, s in enumerate(sentences, 1):
    print(f"{i:02d}: {s}")

# ---------------------------
# 3) TextVectorization: build vocabulary
# ---------------------------
# The TextVectorization layer provides a convenient way to turn raw strings into
# integer token sequences using an internal vocabulary. We will:
#  - instantiate the layer (output_mode='int')
#  - adapt it to the list of sentences (this computes the vocabulary)
#  - use get_vocabulary() to inspect tokens and indices

# Create the TextVectorization layer
# - output_mode='int' means the layer will return integer token IDs for strings
# - we do NOT set output_sequence_length here because we want variable-length output
#   from the layer during inspection. Later we'll pad sequences to a fixed max length.
vectorizer = TextVectorization(output_mode='int')

# Adapt the vectorizer to our sentences (this computes token frequencies and final vocab)
vectorizer.adapt(sentences)

# get_vocabulary() returns a list of the tokens in order. The position in this list
# corresponds to the integer index that the layer produces when called with text.
vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)
print('\nVocabulary size (len(vocab)): ', vocab_size)

# Print a portion of the vocabulary with indices so you can inspect the mapping.
# The mapping is: integer_id -> vocab[integer_id]
print('\nFirst 40 vocabulary items (index: token):')
for idx, token in enumerate(vocab[:40]):
    print(f"{idx:03d}: {repr(token)}")

# NOTE about reserved indices (common TF behaviour):
# - index 0 is typically reserved for the padding token produced by layers that pad
#   sequences. In the returned vocabulary list it may appear as the empty string '' or
#   an explicit padding marker depending on TF version. Inspecting vocab[0] above
#   clarifies what value is stored.
# - index 1 is commonly used for OOV (out-of-vocabulary) token when TextVectorization
#   maps unknown words. Because our vocabulary is built from the corpus, all tokens
#   present in the corpus should map to indices >= 0 and < vocab_size.

# Build convenient mappings for later use
index_to_word = {i: w for i, w in enumerate(vocab)}           # integer id -> token
word_to_index = {w: i for i, w in enumerate(vocab)}           # token -> integer id

# Example: show index of a particular token from the earlier description
sample_word = 'lanigan'  # note: vectorizer standardizes to lowercase by default
sample_index = word_to_index.get(sample_word)
print(f"\nIndex for token '{sample_word}': {sample_index} (None means it's not in vocab)")

# ---------------------------
# 4) Create n-gram training sequences
# ---------------------------
# For each sentence we will create a set of sequences like this (this is a common
# toy approach for teaching next-word prediction):
#   For a sentence with token ids [t1, t2, t3, t4]
#   produce sequences: [t1, t2], [t1, t2, t3], [t1, t2, t3, t4]
#   Each sequence's label is the last token in that sequence (i.e. target next-word).
# Implementation detail: we'll call the vectorizer to convert a sentence to integers,
# strip any padding zeros that may appear, and then generate the n-gram sequences.

all_sequences = []  # will hold lists of integer token ids (variable lengths)

for sentence in sentences:
    # vectorizer returns a 1D int array inside a batch dimension: shape (1, seq_len)
    token_ids = vectorizer([sentence]).numpy()[0]

    # Remove padding zeros (if any) that might be present in the output. For short
    # corpora it's common to get no extra zeros, but this makes our code robust.
    token_ids = token_ids[token_ids != 0]

    # If a sentence has fewer than 2 tokens there's nothing to predict from it; skip.
    if len(token_ids) < 2:
        continue

    # Build n-gram sequences: for i from 1..len-1 build token_ids[:i+1]
    for i in range(1, len(token_ids)):
        ngram = token_ids[: i + 1].tolist()
        all_sequences.append(ngram)

print(f"\nTotal n-gram sequences generated: {len(all_sequences)}")

# Show a few examples with mapping back to human-readable text
print('\nSome n-gram sequences (int ids) and their token forms:')
for seq in all_sequences[:6]:
    print(seq, '->', ' '.join(index_to_word[idx] for idx in seq))

# ---------------------------
# 5) Pad sequences & prepare X / y
# ---------------------------
# We want a single tensor for training. So find the maximum sequence length and pad
# all sequences to that length. We will "pre"-pad sequences so that the last tokens
# line up on the right (this matches the description in your prompt).

max_sequence_len = max(len(s) for s in all_sequences)
print(f"\nMaximum sequence length (including the label token): {max_sequence_len}")

# Pad sequences: shape will be (num_sequences, max_sequence_len)
padded_sequences = pad_sequences(all_sequences, maxlen=max_sequence_len, padding='pre')

# Split into inputs and labels:
# - Inputs X are all tokens except the last token in each padded sequence (so shape
#   will be (num_sequences, max_sequence_len - 1))
# - Labels y are the last token of each sequence
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

print(f"X shape: {X.shape}  (num_examples, sequence_length)")
print(f"y shape: {y.shape}  (num_examples,) - integer labels")

# One-hot encode labels because we'll compile using categorical_crossentropy.
# Explanation: y_one_hot has shape (num_examples, vocab_size). On tiny corpora, this
# is fine. For large vocabularies use sparse_categorical_crossentropy to avoid
# creating a huge one-hot matrix.
num_classes = vocab_size
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=num_classes)
print(f"y_one_hot shape (num_examples, vocab_size): {y_one_hot.shape}")

# Show one example: pick the first sample and explain shapes / values
example_i = 0
print('\nExample (first training sample)')
print('X (token ids):', X[example_i])
print('X (tokens)   :', ' '.join(index_to_word[idx] for idx in X[example_i] if idx != 0))
print('y (int id)   :', y[example_i])
print('y (token)    :', index_to_word[y[example_i]])
print('y one-hot has a 1 at index:', np.where(y_one_hot[example_i] == 1)[0])

# ---------------------------
# 6) Model building
# ---------------------------
# The model used in the description is minimal:
#   - Embedding layer (input_dim=vocab_size, output_dim=64)
#   - Bidirectional LSTM with 20 units
#   - Dense softmax covering the full vocabulary
# Input shape note: the model expects input vectors of length (max_sequence_len - 1).
# Because we pre-padded the inputs, this shape is fixed.

embedding_dim = 64
lstm_units = 20
input_length = X.shape[1]  # equals max_sequence_len - 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_classes, output_dim=embedding_dim, input_length=input_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ---------------------------
# 7) Training
# ---------------------------
# Because the dataset is tiny, we may need many epochs for the model to reach very
# high accuracy on the training set. This is expected -- it simply means the model
# memorized the mapping. For real language modeling, we use vast amounts of data and
# do not expect to reach near-100% training accuracy.

EPOCHS = 200  # you can increase this to 500 as in the example; just be aware of time
history = model.fit(X, y_one_hot, epochs=EPOCHS, verbose=2)

# ---------------------------
# 8) Plot training metrics
# ---------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Training loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('Training accuracy')
plt.xlabel('epoch')

plt.tight_layout()
plt.show()

# ---------------------------
# 9) Prediction & sampling
# ---------------------------
# To generate text we will:
#  1) take a seed sentence (string)
#  2) tokenize it with the SAME vectorizer
#  3) take the last (input_length) tokens as the model input (pre-pad if shorter)
#  4) run the model to get a probability distribution over the vocabulary
#  5) sample from that distribution using a temperature hyperparameter
#  6) append the sampled word to the seed and repeat

# Helper: temperature-based sampling. A temperature <1 sharpens the distribution (more
# greedy), temperature >1 flattens it (more random). A naive greedy strategy would
# be to choose argmax, but that often leads to repetitive output.

def sample_from_probs(preds, temperature=1.0):
    """Convert raw softmax preds into a sampled index using temperature.

    preds: 1D numpy array of probabilities that sum to 1 (shape=(vocab_size,))
    temperature: float > 0. lower -> more conservative, higher -> more random
    """
    preds = np.asarray(preds).astype('float64')
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    # Avoid log(0) by adding a tiny constant
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # Sample a single index from the resulting distribution
    next_index = np.random.choice(range(len(preds)), p=preds)
    return next_index


def generate_text(model, vectorizer, seed_text, next_words=50, temperature=1.0, no_repeat=True):
    """Generate text from seed_text using the trained model and vectorizer.

    Parameters:
    - model: trained keras model that outputs a softmax over the vocabulary
    - vectorizer: TextVectorization layer used to build the vocabulary
    - seed_text: initial text (string)
    - next_words: how many tokens to generate
    - temperature: sampling temperature (float)
    - no_repeat: if True, do a simple check to avoid repeating the same token twice
                 in a row (very simple repetition control)
    """
    output_text = seed_text.strip()

    for _ in range(next_words):
        # Tokenize the current output (the vectorizer returns a padded array in batch)
        tokenized = vectorizer([output_text]).numpy()[0]
        tokenized = tokenized[tokenized != 0]  # remove padding tokens

        # Keep only the rightmost `input_length` tokens for the model input
        tokenized = tokenized[-input_length:]

        # Pad pre to make shape (1, input_length)
        padded = pad_sequences([tokenized], maxlen=input_length, padding='pre')

        # Predict
        preds = model.predict(padded, verbose=0)[0]

        # Optional: forbid the padding-token index if it exists (often index 0)
        # Set its probability to 0 so we never sample a padding token as next word.
        if 0 < len(preds):
            preds[0] = 0.0

        # Sample the next token index using temperature
        next_index = sample_from_probs(preds, temperature=temperature)
        next_word = index_to_word.get(next_index, '')

        # Simple no-repeat rule: if it's the same as previous word, re-sample once
        if no_repeat:
            prev_word = output_text.split()[-1] if len(output_text.split()) > 0 else None
            if next_word == prev_word:
                # reduce the probability of that token and re-sample
                preds[next_index] = 0
                # re-normalize (safe guard: if all zero then fallback to argmax)
                if preds.sum() == 0:
                    next_index = int(np.argmax(preds))
                else:
                    preds = preds / preds.sum()
                    next_index = np.random.choice(range(len(preds)), p=preds)
                    next_word = index_to_word.get(next_index, '')

        # If next_word is a blank or special token, we skip adding it (very small corpora)
        if not next_word or next_word.strip() == '':
            # if we got an empty token, stop generation early
            break

        # Append the sampled word to the output and continue
        output_text += ' ' + next_word

    return output_text

# Example usage of the generator. Replace seed_text with any seed you like.
seed_text = "Lawrence went to Dublin"
print('\nSample generation (greedy-ish with temperature 1.0):')
print(generate_text(model, vectorizer, seed_text, next_words=50, temperature=1.0))

print('\nSample generation (more conservative, temperature 0.5):')
print(generate_text(model, vectorizer, seed_text, next_words=50, temperature=0.5))

# ---------------------------
# 10) Saving artifacts
# ---------------------------
# Save the trained model and the vocabulary so you can reload both later.
model_save_path = 'lanigan_model.keras'
model.save(model_save_path)
print(f"\nModel saved to: {model_save_path}")

vocab_file = 'vocab.txt'
with open(vocab_file, 'w', encoding='utf-8') as f:
    for token in vocab:
        f.write(token + '\n')
print(f"Vocabulary saved to: {vocab_file}")

# To reload the vocabulary later and create a new vectorizer with the same map:
#   with open('vocab.txt') as f:
#       loaded_vocab = [w.strip() for w in f.readlines()]
#   new_vectorizer = TextVectorization(output_mode='int')
#   new_vectorizer.set_vocabulary(loaded_vocab)
# The new_vectorizer will map tokens to the same integer ids as before.

# ---------------------------
# 11) Notes, limitations, and suggestions
# ---------------------------
# - This lab demonstrates how tiny corpora lead to overfitting and memorization. High
#   training accuracy on such a dataset simply means the model learned to reproduce
#   the lines present in the training set. Predictions far from the seed sentence
#   will degrade quickly and may repeat tokens.
# - The one-hot labeling approach used here (categorical_crossentropy with to_categorical)
#   becomes memory-expensive when vocab_size is large. Use sparse_categorical_crossentropy
#   and integer labels (y as ints) for larger corpora:
#       model.compile(loss='sparse_categorical_crossentropy', ...)
#       model.fit(X, y, ...)
# - Experiment with alternative architectures:
#     * increase embedding_dim (e.g., 128) for richer representations
#     * increase lstm_units or stack LSTMs for more capacity
#     * replace BiLSTM with a simpler LSTM if inference speed is important
# - Improved decoding strategies:
#     * beam search for more coherent sequences
#     * nucleus sampling (top-p) or top-k sampling to reduce gibberish
#     * repetition penalties to avoid long repeated words
# - When moving to a corpus with many songs (the planned next step), consider:
#     * using a tokenizer (subword) like WordPiece or SentencePiece to reduce rare
#       token sparsity
#     * switching to Transformer-based models for better long-range coherence
#     * using character-level models if word-level vocabulary becomes too large

# End of lab file. You can now open this file in a Jupyter notebook environment and
# run it cell-by-cell; the comments serve as the explanatory text.
