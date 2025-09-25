# ==========================================================
# Embeddings Extraction and Visualization with TensorFlow
# ==========================================================
# Goal:
# 1. Load IMDb reviews dataset (text + labels).
# 2. Preprocess text with TextVectorization (tokenization + integer mapping).
# 3. Build a neural network with an Embedding layer.
# 4. Train the model for sentiment classification (positive/negative).
# 5. Extract the learned embeddings (word vectors).
# 6. Save embeddings for visualization in TensorFlow Embedding Projector.
# ==========================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# ----------------------------------------------------------
# 1) Load dataset (IMDb reviews dataset)
# ----------------------------------------------------------
# The IMDb dataset contains 50,000 movie reviews labeled as
# positive (1) or negative (0). It's split into 25k train, 25k test.
# Each element is a tuple: (review_text, label).
(train_data, test_data), info = tfds.load(
    "imdb_reviews",          # dataset name
    split=["train", "test"], # request both splits
    as_supervised=True,      # return data as (text, label) pairs
    with_info=True           # return dataset metadata
)

# ----------------------------------------------------------
# 2) Prepare text vectorization layer
# ----------------------------------------------------------
# Why? Neural networks can’t process raw text. We must:
# - Tokenize (split into words).
# - Convert words into integer indices.
# - Pad/truncate reviews to the same length.
# This is done by TextVectorization layer.

SEQUENCE_LENGTH = 120  # Fixed length for each review (truncate/pad)

encoder = tf.keras.layers.TextVectorization(
    max_tokens=10000,             # Build vocab of top 10,000 words
    output_mode="int",            # Convert text → integer sequence
    output_sequence_length=120    # Pad/trim to 120 tokens per review
)

# "Adapt" builds the vocabulary by analyzing training data.
encoder.adapt(train_data.map(lambda text, label: text))

# ----------------------------------------------------------
# 3) Build the model
# ----------------------------------------------------------
# Let's break it down layer by layer:

embedding_dim = 16  # Each word will be mapped to a 16-dim vector

model = tf.keras.Sequential([
    # 1. TextVectorization: raw text → integer sequence
    encoder,
    
    # 2. Embedding Layer:
    # Maps each word index to a 16-dimensional dense vector.
    # Shape: (vocab_size, embedding_dim) = (10000, 16)
    tf.keras.layers.Embedding(10000, embedding_dim),
    
    # 3. GlobalAveragePooling1D:
    # Takes the average across the sequence of word embeddings.
    # Converts variable-length sequences → fixed-size vector.
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # 4. Dense hidden layer:
    # Adds non-linearity and learns abstract patterns.
    tf.keras.layers.Dense(16, activation="relu"),
    
    # 5. Output layer:
    # Single unit with sigmoid → outputs probability [0,1]
    # 0 = negative review, 1 = positive review
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ----------------------------------------------------------
# Model compilation
# ----------------------------------------------------------
# Compile tells TensorFlow HOW to train:
# - Optimizer: "adam" (adaptive learning rate optimizer)
# - Loss: "binary_crossentropy" (because binary classification)
# - Metrics: track "accuracy"
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------------------------------
# 4) Train the model
# ----------------------------------------------------------
# - train_data.batch(512): groups 512 examples per training step
# - epochs=10: iterate 10 times over dataset
# - validation_data=test_data.batch(512): check performance on unseen data
history = model.fit(
    train_data.batch(512),
    epochs=10,
    validation_data=test_data.batch(512),
    verbose=1
)

# ----------------------------------------------------------
# 5) Extract embeddings
# ----------------------------------------------------------
# Every word has a learned vector representation (embedding).
# We fetch those learned weights from the embedding layer.
embedding_layer = model.layers[1]  # Layer index 1 is Embedding
embedding_weights = embedding_layer.get_weights()[0]
print("Embedding shape:", embedding_weights.shape)
# Expected: (10000, 16) → 10k words × 16 dimensions

# ----------------------------------------------------------
# 6) Get vocabulary (words corresponding to each index)
# ----------------------------------------------------------
# encoder.get_vocabulary() returns list of words,
# where index matches embedding_weights row.
vocab = encoder.get_vocabulary()

# ----------------------------------------------------------
# 7) Write embeddings & metadata to files
# ----------------------------------------------------------
# For TensorFlow Embedding Projector:
# - vecs.csv → vectors (embeddings)
# - meta.csv → word labels
import io

out_v = io.open("vecs.csv", "w", encoding="utf-8")
out_m = io.open("meta.csv", "w", encoding="utf-8")

for word_num, word in enumerate(vocab):
    vec = embedding_weights[word_num]  # embedding vector for word
    out_m.write(word + "\n")  # write word to metadata file
    out_v.write(",".join([str(x) for x in vec]) + "\n")  # write numbers

out_v.close()
out_m.close()

print("Files vecs.csv and meta.csv have been created!")

# ----------------------------------------------------------
# 8) Visualization Instructions
# ----------------------------------------------------------
# - Open: https://projector.tensorflow.org/
# - Click "Load data"
# - Upload `vecs.csv` as vectors
# - Upload `meta.csv` as metadata
# - Explore embeddings in 2D/3D, search for words, visualize clusters.
# ----------------------------------------------------------


# ==========================================================
# EXTRA KNOWLEDGE: COMPLETE EXPLANATION
# ==========================================================
# 1. Why embeddings?
#    - Words are categorical, but we need numerical input.
#    - One-hot encoding → too sparse (10,000 words = 10k-dim vector).
#    - Embeddings → dense, low-dimensional representations.
#    - They capture semantic relationships: "king - man + woman ≈ queen".

# 2. How embeddings are learned?
#    - Embedding layer starts with random vectors.
#    - During training, backpropagation adjusts these vectors
#      to minimize loss (classification error).
#    - Words used in similar contexts move closer in vector space.

# 3. GlobalAveragePooling1D explained:
#    - Input shape: (sequence_length, embedding_dim)
#    - Output shape: (embedding_dim,)
#    - Operation: Take mean of each embedding dimension across tokens.
#    Example:
#        "I love cats"
#        Embeddings = [[0.2, 0.8], [0.5, 0.9], [0.6, 0.7]]
#        Average = [(0.2+0.5+0.6)/3, (0.8+0.9+0.7)/3] = [0.43, 0.8]

# 4. Overfitting concept:
#    - If training accuracy >> validation accuracy, model memorizes training data.
#    - Our example: train acc = 1.00, val acc ≈ 0.82 → clear overfitting.
#    - Solutions: dropout, L2 regularization, more data, smaller model.

# 5. Binary Crossentropy loss:
#    - Formula: L = -[y*log(p) + (1-y)*log(1-p)]
#    - y = true label (0/1), p = predicted probability.
#    - Minimizes gap between predicted probabilities and true labels.

# 6. Backpropagation in embeddings:
#    - When a review is processed, only embeddings of words in that review
#      get updated during gradient descent.
#    - Example: "dog" appears in positive review → embedding vector for "dog"
#      shifts toward "positive sentiment space".

# 7. Visualization with TensorFlow Projector:
#    - Projects high-dimensional embeddings (16D) into 2D/3D.
#    - Techniques used: PCA (linear projection) or t-SNE/UMAP (nonlinear).
#    - Helps discover clusters: synonyms group together, opposites separate.

# 8. Why dimension = 16?
#    - Small enough to visualize, large enough to capture meaning.
#    - In real NLP tasks (like BERT), dimensions are often 256–768.

# ==========================================================
# END OF LAB
# After this lab you should know:
# - How text preprocessing (tokenization + encoding) works.
# - How embeddings represent words as dense vectors.
# - How models classify sentiment using embeddings.
# - How to extract, save, and visualize embeddings.
# - Key concepts: embeddings, pooling, loss functions, overfitting.
# ==========================================================
