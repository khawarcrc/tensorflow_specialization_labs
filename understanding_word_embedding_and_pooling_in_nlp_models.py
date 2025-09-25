# ==========================================================
# LAB: Word Embeddings for Sentiment Classification
# ==========================================================
# Goal:
#   - Learn how word embeddings represent words in vector space
#   - Build two models (Flatten vs GlobalAveragePooling1D)
#   - Compare speed, accuracy, and parameter counts
#
# Dataset: IMDB Reviews (50k movie reviews, positive/negative labels)
# Source: Stanford (Andrew Maas et al.)
# Provided by: TensorFlow Datasets (TFDS)
# ==========================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import time

# ---------------------------
# 1) Load IMDB Reviews Dataset
# ---------------------------
# - Each review is text (string), label is integer (0=negative, 1=positive)
# - 25k training reviews, 25k test reviews
# - tfds.load makes downloading, splitting, and preparing easy
(train_ds, test_ds), info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,   # ensures we get (text, label) tuples instead of dicts
    with_info=True        # return metadata about dataset
)

print("IMDB dataset info:", info)

# ---------------------------
# 2) Explore one sample
# ---------------------------
# Each example = (review_text, label)
# - review_text is a tf.Tensor containing bytes (string)
# - label is scalar tf.Tensor (0 or 1)
for text, label in train_ds.take(1):
    print("Sample review (first 300 chars):", text.numpy().decode("utf-8")[:300])
    print("Label (0=neg, 1=pos):", label.numpy())

# ---------------------------
# 3) Text preprocessing setup
# ---------------------------
# Why preprocessing?
# - ML models require numeric input, not raw text
# - We tokenize text (split into words), build vocabulary, map words -> integer IDs
# - To handle variable-length sentences, we pad/truncate them to fixed length
MAX_VOCAB = 10000          # keep top 10,000 most frequent words
OUTPUT_SEQ_LEN = 200       # every review -> exactly 200 tokens (shorter=pad, longer=truncate)
EMBEDDING_DIM = 16         # dimension of embedding space (words -> 16-d vectors)
BATCH_SIZE = 32

# ---------------------------
# 4) Text Vectorization Layer
# ---------------------------
# - Handles tokenization, vocab building, int mapping, and padding
# - Learns vocab from training set only (important to avoid test leakage!)
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=OUTPUT_SEQ_LEN
)

# Adapt vectorizer to training data
#   -> builds vocabulary of top MAX_VOCAB words
train_texts = train_ds.map(lambda x, y: x)  # take only reviews, ignore labels
vectorize_layer.adapt(train_texts)

# Example: check one review after vectorization
example_text = next(iter(train_texts))
print("Vectorized example (first 20 tokens):", vectorize_layer(example_text)[:20])

# ---------------------------
# 5) Preprocessing function
# ---------------------------
# Converts (text, label) -> (token_id_sequence, label)
#   - text is mapped into integer IDs of shape (OUTPUT_SEQ_LEN,)
#   - label is unchanged (0 or 1)
def preprocess_example(text, label):
    seq = vectorize_layer(text)  # tokenize + map + pad
    return seq, label

# Apply preprocessing to datasets
train_ds_proc = train_ds.map(preprocess_example)
test_ds_proc = test_ds.map(preprocess_example)

# Build efficient input pipeline
train_ds_proc = train_ds_proc.shuffle(10000).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
test_ds_proc = test_ds_proc.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# ==========================================================
# Embeddings: Intuition
# ==========================================================
# - Each word is represented as a trainable vector (embedding) of size EMBEDDING_DIM.
# - Words with similar meaning or context (e.g., "boring", "dull") get similar vectors.
# - During training, embeddings shift so that sentiment-related clusters emerge.
# - Example: Negative words cluster near each other, positive words cluster together.
# - Embedding output per review = (seq_len, emb_dim) -> e.g., (200, 16)
# ==========================================================

# -------------------------------------------------------
# 6) Model A: Using Flatten
# -------------------------------------------------------
# Process:
#   Input -> Embedding -> Flatten -> Dense -> Dense
# Key Idea:
#   - Flatten reshapes (seq_len, emb_dim) -> (seq_len * emb_dim)
#   - This keeps positional information, but leads to huge feature vectors
#   - More parameters, slightly slower
def make_model_flatten(vocab_size=MAX_VOCAB, emb_dim=EMBEDDING_DIM, seq_len=OUTPUT_SEQ_LEN):
    inputs = tf.keras.Input(shape=(seq_len,), dtype="int32", name="input_tokens")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        input_length=seq_len,
        name="embedding"
    )(inputs)                          # shape: (batch, seq_len, emb_dim)
    x = tf.keras.layers.Flatten(name="flatten")(x)  # shape: (batch, seq_len * emb_dim)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)  # regularization to avoid overfitting
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name="model_flatten")
    return model

# -------------------------------------------------------
# 7) Model B: Using GlobalAveragePooling1D
# -------------------------------------------------------
# Process:
#   Input -> Embedding -> GlobalAveragePooling1D -> Dense -> Dense
# Key Idea:
#   - GAP averages across sequence axis: (seq_len, emb_dim) -> (emb_dim,)
#   - Compresses each sentence into one fixed-size vector
#   - Much smaller parameter count, faster to train
#   - Loses positional info but often works well for classification
def make_model_gap(vocab_size=MAX_VOCAB, emb_dim=EMBEDDING_DIM, seq_len=OUTPUT_SEQ_LEN):
    inputs = tf.keras.Input(shape=(seq_len,), dtype="int32", name="input_tokens")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_dim,
        input_length=seq_len,
        name="embedding"
    )(inputs)                           # shape: (batch, seq_len, emb_dim)
    x = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)  # -> (batch, emb_dim)
    x = tf.keras.layers.Dense(16, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs, name="model_gap")
    return model

# ---------------------------
# 8) Compile both models
# ---------------------------
# - Use same optimizer/loss/metrics to fairly compare
model_flatten = make_model_flatten()
model_gap = make_model_gap()

for m in (model_flatten, model_gap):
    m.compile(
        optimizer="adam",                  # adaptive learning optimizer
        loss="binary_crossentropy",        # suitable for binary classification
        metrics=["accuracy"]               # track accuracy during training
    )

print("=== Flatten model summary ===")
model_flatten.summary()
print("\n=== GlobalAveragePooling1D model summary ===")
model_gap.summary()

# ---------------------------
# 9) Train & Compare
# ---------------------------
# - Run both models for 10 epochs
# - Measure training time per epoch
# - Observe accuracy differences
EPOCHS = 10

# GAP model training
start = time.time()
history_gap = model_gap.fit(
    train_ds_proc,
    validation_data=test_ds_proc,
    epochs=EPOCHS
)
end = time.time()
print(f"GAP model: {EPOCHS} epochs took {(end-start)/EPOCHS:.3f} sec/epoch")

# Flatten model training
start = time.time()
history_flatten = model_flatten.fit(
    train_ds_proc,
    validation_data=test_ds_proc,
    epochs=EPOCHS
)
end = time.time()
print(f"Flatten model: {EPOCHS} epochs took {(end-start)/EPOCHS:.3f} sec/epoch")

# ---------------------------
# 10) Final Results
# ---------------------------
print("GAP final training accuracy:", history_gap.history["accuracy"][-1])
print("GAP final validation accuracy:", history_gap.history["val_accuracy"][-1])
print("Flatten final training accuracy:", history_flatten.history["accuracy"][-1])
print("Flatten final validation accuracy:", history_flatten.history["val_accuracy"][-1])

# ==========================================================
# Observations:
# ----------------------------------------------------------
# - Flatten model: More parameters, slightly slower, may reach higher training accuracy
#   (risk of overfitting if validation accuracy lags behind).
# - GAP model: Fewer parameters, faster, simpler. Accuracy is often comparable.
# - Both approaches are valid; choice depends on trade-off between speed and accuracy.
#
# In the example:
# - GAP ~0.82 validation accuracy, ~6.2 sec/epoch
# - Flatten ~0.83 validation accuracy, ~6.5 sec/epoch
#
# Try it yourself with different embedding sizes, sequence lengths, or pooling layers
# (e.g., GlobalMaxPooling1D, LSTM, GRU, Conv1D).
# ==========================================================
