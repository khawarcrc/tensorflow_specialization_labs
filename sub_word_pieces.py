# =============================================================
# Problem statement
# =============================================================
# Build a reproducible lab that:
# 1. Trains a WordPiece (subword) vocabulary on the IMDB reviews dataset.
# 2. Instantiates a KerasNLP WordPiece tokenizer from that vocabulary.
# 3. Uses the tokenizer to convert raw text to integer token sequences.
# 4. Trains a simple classification model (Embedding -> GlobalAveragePooling1D -> Dense)
#    to predict sentiment (positive/negative) on the IMDB dataset.
# 5. Demonstrates tokenization, detokenization (where useful), saving the vocab,
#    and plotting training curves.
#
# Requirements: Code should be runnable in Google Colab or as a standalone Python script.
# See the "INSTALL" block below for pip install instructions that will run automatically
# if packages are missing.
#
# =============================================================
# Execution theory (high-level explanation)
# =============================================================
# 1) Why subword tokenization?  
#    Subword tokenization (WordPiece/BPE) splits words into smaller units. This helps the
#    model handle unknown words, reduce vocabulary size, and capture meaningful morphemes
#    like prefixes/suffixes. For example, "tokenization" -> ["token", "##ization"].
#
# 2) How the WordPiece vocabulary is created
#    We use KerasNLP's utility compute_word_piece_vocabulary which scans a corpus of
#    text (here: IMDB training reviews) and produces a ranked list of tokens (subwords).
#    You can set the target vocabulary size (e.g. 8,000) and reserved tokens (e.g. [PAD],[UNK]).
#
# 3) Tokenizer behavior in this lab
#    The WordPieceTokenizer when created with sequence_length will produce fixed-length
#    integer sequences (truncating or padding as needed). Those integers index into the
#    vocabulary and are ready for an Embedding layer.
#
# 4) Model architecture and why GlobalAveragePooling1D
#    Because token sequences come out as [batch_size, seq_len] and the embedding produces
#    [batch_size, seq_len, embedding_dim], using GlobalAveragePooling1D collapses the
#    sequence dimension to a single vector by averaging token embeddings. This is simple,
#    efficient, and works well for baseline sentiment tasks like IMDB.
#
# 5) End-to-end notes
#    We will build a tf.data pipeline that batches text first, then tokenizes (so the
#    tokenizer gets a batch of strings and returns a batch of token ids). This makes the
#    pipeline efficient and compatible with the tokenizer Keras layer.
#
# =============================================================
# INSTALL: (will attempt to auto-install missing packages)
# =============================================================
import sys
import subprocess
import importlib

# List of required packages. In Colab you can also run the commented !pip command.
REQUIRED_PACKAGES = [
    "tensorflow",           # tensorflow (2.x) for training, tf.data, layers
    "tensorflow-datasets",  # tfds for IMDB dataset loader
    "keras-nlp",            # keras_nlp for WordPiece tokenizer and vocabulary utility
    "matplotlib",           # plotting training curves
    "numpy",
]

# Attempt to import each package; if it fails, install it with pip.
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg.replace("-", "_"))
    except Exception:
        print(f"Package '{pkg}' not found. Installing... (this may take a minute)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# (Optional) If running in Google Colab, you can uncomment and run the following in a cell:
# !pip install -q tensorflow keras-nlp tensorflow-datasets matplotlib

# =============================================================
# Imports (now that installations were attempted)
# =============================================================
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_nlp
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# Config: hyperparameters and paths
# =============================================================
VOCAB_SIZE = 8000                 # maximum number of subword tokens to keep
RESERVED_TOKENS = ["[PAD]", "[UNK]"]  # indexable reserved tokens
SEQUENCE_LENGTH = 256             # fixed sequence length for tokenization
BATCH_SIZE = 64
EMBED_DIM = 128
EPOCHS = 3                        # small number for demo; raise for better results

# Optional: file path to save vocab
VOCAB_SAVE_PATH = "wordpiece_vocab.txt"

# Reproducibility (not guaranteed across platforms but helps)
tf.random.set_seed(42)
np.random.seed(42)

# =============================================================
# 1) Load IMDB dataset using TFDS
#    We load as_supervised=True which yields (text, label) pairs
# =============================================================
print("Loading IMDB dataset from TFDS...")
(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
)

print("Dataset loaded. Dataset info summary:")
print(" - Description: ", ds_info.description.splitlines()[0])
print(" - Features: ", ds_info.features)

# =============================================================
# 2) Build/train a WordPiece vocabulary from the training text
#    compute_word_piece_vocabulary expects a dataset (or list of filenames).
#    We'll pass a batched dataset of raw strings to make it efficient.
# =============================================================
# Extract only the text (ignore labels) and batch to large chunks for vocabulary training
train_text_ds = ds_train.map(lambda text, label: text)
# Batch into large batches so the vocabulary builder can inspect more text per step
train_text_ds_batched = train_text_ds.batch(4096)

print("Training a WordPiece vocabulary (this may take a little while)...")
# compute_word_piece_vocabulary will scan the batched strings and return a list of tokens
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    train_text_ds_batched,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,                  # lowercase for normalization
    reserved_tokens=RESERVED_TOKENS, # keep pad and unk tokens at predictable indices
)

print(f"Vocabulary trained. Total tokens returned: {len(vocab)} (requested {VOCAB_SIZE})")
print("Sample tokens (first 50):", vocab[:50])

# (Optional) Save vocabulary to disk so you can reuse it later
with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")
print(f"Saved vocabulary to: {VOCAB_SAVE_PATH}")

# =============================================================
# 3) Instantiate the WordPiece tokenizer with the learned vocab
#    We pass sequence_length so the tokenizer returns fixed-size integer sequences.
# =============================================================
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQUENCE_LENGTH,
    lowercase=True,
)

print("Tokenizer created. Vocabulary size according to tokenizer:", tokenizer.vocabulary_size())

# =============================================================
# 4) Show tokenization on a sample string and explain tokens
# =============================================================
sample_text = "I loved this movie! The acting was great, but the plot was a little slow."
print("Sample text:\n", sample_text)
# Tokenize a single sample (tokenizer expects batch input, so wrap in a list)
sample_token_ids = tokenizer([sample_text])  # shape: (1, SEQUENCE_LENGTH)
# Convert tensor to numpy and squeeze batch dimension
sample_token_ids = tf.squeeze(sample_token_ids, axis=0).numpy()

# Get vocabulary list from tokenizer to map ids back to token strings
vocab_list = tokenizer.get_vocabulary()

# Find the PAD token id if present and trim trailing pads for display
pad_id = tokenizer.token_to_id(RESERVED_TOKENS[0]) if RESERVED_TOKENS[0] in vocab_list else None

# Build a readable token list for the first N tokens (skip pads at the end)
N = 60
readable_tokens = []
for idx in sample_token_ids[:N]:
    readable_tokens.append(vocab_list[int(idx)])

print("Token IDs (first 60):", sample_token_ids[:N].tolist())
print("Tokens (first 60):", readable_tokens)

# Notice: tokens with '##' (or the suffix_indicator) indicate subword pieces that are not
# starting at a word boundary. The exact marker used can vary (e.g. '##' or another symbol).

# =============================================================
# 5) Build tf.data pipelines that produce (token_ids, label) batches for training
#    We batch first, then call the tokenizer (a Keras layer) inside map(). This is efficient
#    because the tokenizer is optimized to process batches.
# =============================================================
AUTOTUNE = tf.data.AUTOTUNE

train_ds_prepared = (
    ds_train
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .map(lambda texts, labels: (tokenizer(texts), labels), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds_prepared = (
    ds_test
    .batch(BATCH_SIZE)
    .map(lambda texts, labels: (tokenizer(texts), labels), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# =============================================================
# 6) Build the model
#    Input: integer token IDs shaped (SEQUENCE_LENGTH,)
#    Embedding: maps token ids -> vectors
#    GlobalAveragePooling1D: averages across the sequence dimension to get a fixed vector
#    Dense layers: classification head -> output probability for positive sentiment
# =============================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name="input_tokens"),
    tf.keras.layers.Embedding(
        input_dim=tokenizer.vocabulary_size(),  # must match tokenizer vocabulary size
        output_dim=EMBED_DIM,
        name="token_embedding",
    ),
    tf.keras.layers.GlobalAveragePooling1D(name="avg_pool"),
    tf.keras.layers.Dense(64, activation="relu", name="dense_1"),
    tf.keras.layers.Dropout(0.5, name="dropout"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

print("Model summary:")
model.summary()

# =============================================================
# 7) Train the model
# =============================================================
history = model.fit(
    train_ds_prepared,
    validation_data=val_ds_prepared,
    epochs=EPOCHS,
)

# =============================================================
# 8) Plot training & validation curves (loss and accuracy)
# =============================================================
plt.figure(figsize=(12, 4))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history.get("accuracy", []), label="train_acc")
plt.plot(history.history.get("val_accuracy", []), label="val_acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# =============================================================
# 9) Evaluate on test set and demonstrate inference on custom examples
# =============================================================
print("Evaluating on the test set...")
results = model.evaluate(val_ds_prepared)
print("Test loss & accuracy:", results)

# Inference: function that accepts raw strings and returns probability
def predict_sentiment(texts):
    """Given a list of raw strings, return model probabilities for positive sentiment.
    This function shows how to run the tokenizer + model end-to-end for inference.
    """
    # Tokenizer accepts a batch of strings, returns integer sequences of shape (batch, SEQUENCE_LENGTH)
    tokenized = tokenizer(texts)
    probs = model.predict(tokenized)
    return probs.flatten()

examples = [
    "What a fantastic movie — I loved every second of it!",
    "Terrible film. The plot made no sense and it was boring.",
]
probs = predict_sentiment(examples)
for text, p in zip(examples, probs):
    print(f"\nText: {text}\nPredicted positive probability: {p:.4f} ->", "POSITIVE" if p > 0.5 else "NEGATIVE")

# =============================================================
# End of lab. Tips & next steps:
#  - Try raising VOCAB_SIZE and SEQUENCE_LENGTH to see if performance improves.
#  - Replace GlobalAveragePooling1D with a simple LSTM/Conv1D or a Transformer encoder for
#    sequence models that capture order and longer dependencies.
#  - Save the tokenizer vocabulary and the model for later inference in production.
# =============================================================
