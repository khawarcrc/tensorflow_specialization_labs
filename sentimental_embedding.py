"""
IMDB Sentiment Embeddings Lab
=================================

Problem statement
-----------------
Build an end-to-end lab that:
1. Loads the IMDB reviews dataset using tensorflow_datasets (tfds).
2. Prints dataset metadata so the student can inspect description/features/citation.
3. Builds a text preprocessing pipeline (vocabulary building / tokenization / padding).
4. Trains a small embedding-based neural network for binary sentiment classification.
5. Exports the learned embedding vectors and a metadata file that can be loaded into the
   TensorBoard Embedding Projector (or the online projector) to visualize word clusters.

Execution theory (high-level)
-----------------------------
1. We use tfds to fetch IMDB reviews and metadata. The dataset contains splits: 'train',
   'test' (and an 'unsupervised' split we won't use here).
2. We build a Keras TextVectorization layer to learn the vocabulary from the training split
   only. This guarantees we don't leak information from test/validation when building
   token indices.
3. TextVectorization converts raw strings into integer sequences. We configure it so that
   sequences are padded / truncated to a fixed length (MAX_LEN) so the network receives
   fixed-size input.
4. A small Embedding layer (VOCAB_SIZE x EMBEDDING_DIM) learns vector representations
   for words during supervised training. We flatten the embedding output and feed it into
   a small feed-forward classifier for binary sentiment.
5. After training, we extract the embedding matrix and the vocabulary and write two files
   (vecs.tsv and meta.tsv) suitable for the Embedding Projector.

How to use this file
--------------------
1. Make sure you have TensorFlow and TensorFlow Datasets installed (see REQUIREMENTS).
2. Run the script: `python IMDB_Embeddings_Complete_Lab.py`
3. After training, find `vecs.tsv` and `meta.tsv` in the working directory and upload
   them to the TensorBoard Embedding Projector or the online projector.

REQUIREMENTS
------------
pip install tensorflow tensorflow-datasets

Notes & tips
------------
- This lab is written to be clear and educational rather than maximally optimized.
- For production or faster iteration, use cached datasets and consider distributed training.
- If you want a Jupyter-friendly version, wrap calls into cells; this script is runnable as-is.

"""

# ------------------------------
# Imports and basic settings
# ------------------------------
import os                                     # Operating system utilities (paths, file ops)
import sys                                    # System utilities (exit, version info)
import textwrap                               # For nicer printing of long metadata

import numpy as np                            # Numerical utilities (used when saving vectors)
import tensorflow as tf                       # TensorFlow core library
import tensorflow_datasets as tfds            # TFDS: easy dataset loading / info

# Set a seed for reproducibility where possible
SEED = 42
tf.random.set_seed(SEED)                      # Set TF random seed
np.random.seed(SEED)                          # Set NumPy random seed

# ------------------------------
# Hyperparameters & constants
# ------------------------------
VOCAB_SIZE = 10000     # we'll keep the top 10,000 tokens from training data
MAX_LEN = 120          # maximum number of tokens per review (pad/truncate to this length)
EMBEDDING_DIM = 16     # dimension of the learned word embeddings
BATCH_SIZE = 32        # batch size for training
EPOCHS = 5             # number of training epochs (small for demo)
BUFFER_SIZE = 10000    # buffer size for shuffling

# Path to write embedding projector files
OUTPUT_DIR = os.path.abspath('.')            # write files to current directory by default
VECTORS_TSV = os.path.join(OUTPUT_DIR, 'vecs.tsv')
META_TSV = os.path.join(OUTPUT_DIR, 'meta.tsv')

# ------------------------------
# 1) Load dataset + print info
# ------------------------------
# tfds.load with with_info=True returns a tuple (datasets, info) when split is not
# specified. Datasets will be a dictionary mapping split names to tf.data.Dataset objects.
print('\nLoading IMDB reviews dataset via tfds... (this may download the data)')

# Load the dataset as a dictionary of splits and request metadata information.
# as_supervised=True returns (text, label) pairs instead of a dict per example.
imdb_data, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# `imdb_data` is a dict: keys typically include 'train', 'test' and maybe 'unsupervised'.
train_ds = imdb_data['train']                 # training split as tf.data.Dataset
test_ds = imdb_data['test']                   # test split as tf.data.Dataset

# Print a compact part of the dataset metadata to help learners inspect it.
print('\n=== Dataset info summary ===')
print('Dataset name: imdb_reviews')
print('Description (truncated):')
print(textwrap.fill(info.description.strip().split('\n')[0][:500] + '...', width=80))
print('\nFeatures:')
print(info.features)                           # structure of features (text, label)
print('\nCitation:')
print(info.citation)                           # show the citation for the dataset
print('\nSupervised keys:', info.supervised_keys)  # keys when as_supervised=True

# Show 2 example reviews from the training set so users see the raw format
print('\nTwo sample training examples (truncated to 200 chars):')
for i, (text, label) in enumerate(train_ds.take(2)):
    # `text` is a tf.Tensor with dtype string. Call .numpy() and decode to print clean text.
    decoded = text.numpy().decode('utf-8')    # convert bytes to string
    print(f'  Example {i+1} (label={label.numpy()}):')
    print('    ' + decoded[:200].replace('\n', ' ') + ('...' if len(decoded) > 200 else ''))

# ------------------------------
# 2) Build tokenization / vectorization layer
# ------------------------------
# We'll use Keras's TextVectorization layer which will:
#  - learn the top VOCAB_SIZE tokens from the training sentences
#  - map strings to integer token indices
#  - automatically pad/truncate if we pass output_sequence_length

# Create the TextVectorization layer. The output_sequence_length argument forces
# fixed-size integer sequences (good for simple sequential models).
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,                   # keep only top VOCAB_SIZE tokens
    output_mode='int',                       # integer token indices
    output_sequence_length=MAX_LEN           # pad/truncate every example to MAX_LEN
)

# We must adapt the vectorization only on the training sentences so the vocabulary
# does not see the test set. Use `.map(lambda x,y: x)` to extract strings from pairs.
print('\nAdapting TextVectorization to training data (this builds the vocab).')
# batching here speeds up the adapt process (vectorize_layer.adapt accepts a dataset or a numpy array)
vectorize_layer.adapt(train_ds.map(lambda x, y: x).batch(1024))

# Optional: inspect the first 20 vocabulary entries
vocab = vectorize_layer.get_vocabulary()
print('\nSample of learned vocabulary (first 20 tokens):')
print(vocab[:20])

# ------------------------------
# 3) Prepare tf.data pipelines for training and testing
# ------------------------------
# Vectorize each text and keep the label. Then cache, shuffle and batch for the train set.
# Note: vectorize_layer returns int tensors of shape (MAX_LEN,) because we passed
# output_sequence_length earlier.

# Create function to transform (text, label) -> (vectorized_text, label)
def vectorize_text(text, label):
    """Map function for tf.data to vectorize raw text using the fitted layer.

    Args:
        text: tf.Tensor of dtype string (single example text)
        label: tf.Tensor integer label (0 or 1)
    Returns:
        (vectorized_text, label) where vectorized_text is an int32 tensor shape=(MAX_LEN,)
    """
    text = tf.expand_dims(text, -1) if tf.rank(text) == 0 else text
    # The vectorize_layer accepts a batch of strings or a single string tensor.
    vectorized = vectorize_layer(text)
    return vectorized, label

# Apply the mapping to the datasets. Use AUTOTUNE where helpful for performance.
train_ds = train_ds.map(lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache()                    # cache in memory for faster epoch loops
train_ds = train_ds.shuffle(BUFFER_SIZE, seed=SEED)  # randomize order of examples
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # batch and prefetch

# For the test dataset, we vectorize but do not shuffle or cache in the same way.
# We still batch and prefetch for efficient evaluation.
test_ds = test_ds.map(lambda x, y: (vectorize_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ------------------------------
# 4) Build the model (Embedding -> Flatten -> Dense -> Output)
# ------------------------------
# Simple sequential model implementing the architecture described in the lab text.
model = tf.keras.Sequential([
    # Embedding layer: maps integer tokens to EMBEDDING_DIM-dimensional dense vectors.
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    # Flatten the sequence of vectors into a single 1D vector per example.
    tf.keras.layers.Flatten(),
    # Small dense hidden layer with ReLU activation to allow more modeling capacity.
    tf.keras.layers.Dense(6, activation='relu'),
    # Output layer with a single neuron and sigmoid activation for binary classification.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model specifying optimizer, loss function, and metrics.
model.compile(
    optimizer=tf.keras.optimizers.Adam(),       # adaptive optimizer
    loss='binary_crossentropy',                 # appropriate for binary labels + sigmoid output
    metrics=['accuracy']                        # track accuracy during training
)

# Print a model summary so learners can inspect layers and parameter counts.
print('\nModel summary:')
model.summary()

# ------------------------------
# 5) Train the model
# ------------------------------
print(f'\nTraining for {EPOCHS} epochs (this may take a few minutes depending on your machine).')
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

# ------------------------------
# 6) Evaluate the model on the test set
# ------------------------------
print('\nEvaluating on the test set:')
results = model.evaluate(test_ds)
print('Test results (loss, accuracy):', results)

# ------------------------------
# 7) Export embeddings + metadata for Embedding Projector
# ------------------------------
# Extract the weights from the embedding layer. The embedding matrix shape is
# (input_dim, EMBEDDING_DIM). Each row corresponds to a token index.
embedding_layer = model.layers[0]               # first layer is the Embedding
embedding_weights = embedding_layer.get_weights()[0]  # numpy array shape (VOCAB_SIZE, EMBEDDING_DIM)

# Prepare and write the meta and vector files in TSV format (projector expects this form):
print('\nWriting embedding vectors and metadata for the projector...')

# The vocabulary list returned by vectorize_layer.get_vocabulary() maps token index -> token string.
vocabulary = vectorize_layer.get_vocabulary()

# Ensure the mapping sizes are compatible. If there is a mismatch, we take the minimum length
# to avoid indexing errors. Usually, len(vocabulary) == embedding_weights.shape[0] == VOCAB_SIZE.
num_tokens = min(len(vocabulary), embedding_weights.shape[0])

with open(META_TSV, 'w', encoding='utf-8') as meta_f, open(VECTORS_TSV, 'w', encoding='utf-8') as vec_f:
    # Iterate token indices and write the token and its vector to the files
    for i in range(num_tokens):
        token = vocabulary[i]
        # Write the token (word) to the meta file; the projector will show this label
        meta_f.write(token + '\n')
        # Convert the embedding row to a tab-separated string and write to vecs.tsv
        vector = embedding_weights[i]
        vec_f.write('\t'.join([str(x) for x in vector]) + '\n')

print(f'Wrote {num_tokens} tokens to:')
print('  ', META_TSV)
print('  ', VECTORS_TSV)

# ------------------------------
# Additional suggestions (not executed) - experiment/hyperparameters
# ------------------------------
# 1) Use GlobalAveragePooling1D instead of Flatten to get a more robust sentence
#    representation that is less sensitive to length and order.
# 2) Add Dropout to reduce overfitting: tf.keras.layers.Dropout(0.5)
# 3) Increase EMBEDDING_DIM to 50 or 100 for richer representations, at the cost of
#    more parameters and longer training times.
# 4) To inspect nearest neighbors in the embedding space programmatically, compute
#    cosine similarity between embedding vectors and look at top-k indices.

# Example: compute the nearest neighbors for a target word (uncomment to use)
#
# from sklearn.metrics.pairwise import cosine_similarity
# def nearest_neighbors(word, top_k=10):
#     if word not in vocabulary:
#         print('Word not in vocabulary')
#         return
#     idx = vocabulary.index(word)
#     emb = embedding_weights[idx].reshape(1, -1)
#     sims = cosine_similarity(embedding_weights, emb).reshape(-1)
#     nearest = sims.argsort()[::-1][1:top_k+1]
#     return [(vocabulary[i], sims[i]) for i in nearest]
#
# print('Nearest to "good":', nearest_neighbors('good', top_k=10))

print('\nLab complete. You can open the vecs.tsv and meta.tsv files in the TensorBoard\nEmbedding Projector or upload them to the online projector at projector.tensorflow.org.')

# End of script
