# ==========================================================
# Lab: Preparing Training Data for Next-Word Prediction
# ==========================================================
# Goal:
#   1. Convert a text corpus into training data for a next-word prediction model.
#   2. Understand how to tokenize, build n-gram sequences, pad them, and split into X (inputs) and y (labels).
#   3. Train a simple LSTM model to predict the next word given a prefix.
# ==========================================================

# ------------------------------
# 1. Import Required Libraries
# ------------------------------
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# ------------------------------
# 2. Define a Small Corpus
# ------------------------------
# Each line here is treated as a separate sentence.
# In real projects, this could be thousands of lines (e.g., song lyrics, poems, articles).
corpus = [
    "in the town of Athy one Jeremy Lanigan",
    "there lived a man",
    "he had a very large family",
    "and he never was lazy"
]

# ------------------------------
# 3. Tokenization
# ------------------------------
# Tokenizer builds a mapping from each unique word -> integer index.
# oov_token="<OOV>" ensures that unseen words map to a safe placeholder.
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)   # learn word-index mapping

# Vocabulary size = number of unique tokens + 1 for padding (0 is reserved)
vocab_size = len(tokenizer.word_index) + 1

print("Word Index Mapping:", tokenizer.word_index)
print("Vocabulary Size:", vocab_size)

# ------------------------------
# 4. Generate n-gram sequences
# ------------------------------
# For each sentence, break it into prefixes of increasing length.
# Example: "in the town of" -> tokens [1,2,3,4]
#   Prefixes: [1,2], [1,2,3], [1,2,3,4]
sequences = []

for line in corpus:
    # Convert sentence to list of token IDs
    token_list = tokenizer.texts_to_sequences([line])[0]

    # Generate n-gram sequences starting from length 2
    for i in range(2, len(token_list) + 1):
        ngram = token_list[:i]
        sequences.append(ngram)

print("\nGenerated Sequences (before padding):")
print(sequences)

# ------------------------------
# 5. Find the maximum sequence length
# ------------------------------
# Needed so we can pad all sequences to the same length.
maxlen = max(len(s) for s in sequences)
print("\nMaximum sequence length:", maxlen)

# ------------------------------
# 6. Pre-pad all sequences with zeros
# ------------------------------
# Padding ensures all sequences are the same length.
# We pad on the left ("pre") so that the *last* token is always the true next-word label.
padded = pad_sequences(sequences, maxlen=maxlen, padding='pre')

print("\nPadded Sequences:")
print(padded)

# ------------------------------
# 7. Split into X (inputs) and y (labels)
# ------------------------------
# For each padded sequence:
#   - Input (X) = all tokens except the last
#   - Label (y) = the last token
X = padded[:, :-1]
y = padded[:, -1]

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Sample X[0]:", X[0])
print("Sample y[0]:", y[0])

# ------------------------------
# 8. Build a Simple LSTM Model
# ------------------------------
# This is a minimal model for demonstration.
# Embedding layer learns vector representations for each token.
# LSTM learns sequential dependencies.
# Dense output layer predicts probability distribution over vocabulary.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64,
                              input_length=X.shape[1], mask_zero=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ------------------------------
# 9. Train the Model
# ------------------------------
# Normally you’d need thousands of examples; here it's just a demo.
# y is sparse (integers), so we use sparse_categorical_crossentropy.
history = model.fit(X, y, epochs=50, verbose=0)

print("\nTraining complete!")

# ------------------------------
# 10. Generate Text (Optional Demo)
# ------------------------------
# Start with a seed phrase and let the model predict next words.
def generate_text(seed_text, next_words=5):
    for _ in range(next_words):
        # Convert current seed text to sequence of tokens
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad to match input length
        token_list = pad_sequences([token_list], maxlen=X.shape[1], padding='pre')
        # Predict next word
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

        # Map back from token -> word
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text

print("\nGenerated text from seed 'in the':")
print(generate_text("in the", next_words=5))

# ==========================================================
# Key Takeaways:
# 1. Each sentence is converted into many prefix sequences.
# 2. Pre-padding makes label extraction trivial (last token is always the label).
# 3. X = all but last token, y = last token.
# 4. Model can now learn next-word prediction via supervised training.
# ==========================================================
