# ----------------------------------------------------------
# Import required libraries
# ----------------------------------------------------------
import tensorflow as tf                     # TensorFlow: main ML/DL framework
import tensorflow_datasets as tfds          # TensorFlow Datasets: provides ready-to-use datasets

# ----------------------------------------------------------
# 1. Load IMDB dataset
# ----------------------------------------------------------
# tfds.load() will:
#   1. Download the dataset (if not already cached locally).
#   2. Prepare the dataset into a standard TensorFlow format.
#   3. Return splits (train/test/validation).
# Parameters:
#   "imdb_reviews"   -> dataset name (string).
#   split=["train","test"] -> which splits to return.
#   as_supervised=True     -> return (input, label) pairs directly, instead of dicts.
#   with_info=True         -> also return metadata (class names, number of samples, etc.).
(train_data, test_data), info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset info:", info)   # prints dataset metadata (num_examples, features, etc.)

# ----------------------------------------------------------
# 2. Explore a single data point
# ----------------------------------------------------------
# train_data is a tf.data.Dataset object containing (review, label) pairs.
# The `.take(1)` method pulls out 1 sample from the dataset.
# Each sample is a tuple: (review_text, label)
for review, label in train_data.take(1):
    # review is a tf.Tensor holding raw text (bytes string).
    # .numpy() converts tf.Tensor → NumPy array.
    # .decode("utf-8") converts byte string → readable Python string.
    print("Review text:", review.numpy().decode("utf-8"))
    
    # label is a tf.Tensor with value 0 or 1.
    # .numpy() converts it to an integer we can print easily.
    print("Label:", label.numpy())  # 1 = positive, 0 = negative

# ----------------------------------------------------------
# 3. Split reviews and labels
# ----------------------------------------------------------
# The dataset contains (review, label) pairs.
# Sometimes we want to separate them into just reviews or just labels.
# We can do this using .map(), which applies a function to every element.

# Extract only reviews (x).
train_sentences = train_data.map(lambda x, y: x)
train_labels = train_data.map(lambda x, y: y)

# Do the same for test set.
test_sentences = test_data.map(lambda x, y: x)
test_labels = test_data.map(lambda x, y: y)

# Now we have four datasets: reviews (train/test) and labels (train/test).

# ----------------------------------------------------------
# 4. Text Vectorization Layer
# ----------------------------------------------------------
# Machine learning models cannot directly understand text → must convert words to numbers.
# The TextVectorization layer does three things:
#   1. Build a vocabulary (map words → integer IDs).
#   2. Convert input text → integer sequences.
#   3. Optionally pad/truncate sequences to a fixed length.

MAX_TOKENS = 10000          # Only keep the 10,000 most frequent words in the vocab.
OUTPUT_SEQUENCE_LENGTH = 200 # After tokenization, pad/truncate every review to length 200.

# Create the vectorization layer with given parameters.
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_sequence_length=OUTPUT_SEQUENCE_LENGTH
)

# Adapt the vectorizer on training data → learn which words are common.
# This scans all training reviews, builds vocabulary, and sets up the internal mapping.
vectorizer.adapt(train_sentences)

# Example usage: vectorize one review.
# next(iter(...)) gets the first element from train_sentences.
example_review = next(iter(train_sentences))
# Vectorizer automatically tokenizes, maps to integers, and pads.
print("Vectorized example:", vectorizer(example_review))

# ----------------------------------------------------------
# 5. Padding Function (wrap vectorization + padding)
# ----------------------------------------------------------
def preprocess_text(text, label):
    """
    Function to vectorize and pad text.
    Args:
        text (tf.Tensor): Input review sentence.
        label (tf.Tensor): Corresponding label (0 = negative, 1 = positive).
    Returns:
        vectorized_text (tf.Tensor): Integer sequence (length = OUTPUT_SEQUENCE_LENGTH).
        label (tf.Tensor): Same label, unchanged.
    """
    text = vectorizer(text)  # Apply vectorization & automatic padding/truncation.
    return text, label       # Return processed text along with its label.

# Apply preprocessing to both training and test datasets.
# This step transforms each review from string → integer sequence.
train_ds = train_data.map(preprocess_text)
test_ds = test_data.map(preprocess_text)

# ----------------------------------------------------------
# 6. Prepare Final Dataset Pipeline
# ----------------------------------------------------------
BATCH_SIZE = 32  # Number of samples per training step (mini-batch size).

# Prepare training dataset pipeline
train_ds = (
    train_ds
    .shuffle(buffer_size=10000)  # Shuffle dataset to avoid training order bias.
    .cache()                     # Cache data in memory for faster reuse.
    .prefetch(tf.data.AUTOTUNE)  # Overlap preprocessing with model training.
    .batch(BATCH_SIZE)           # Group samples into batches of 32.
)

# Prepare test dataset pipeline
test_ds = (
    test_ds
    .cache()                     # No shuffle needed (order doesn’t matter for testing).
    .prefetch(tf.data.AUTOTUNE)  # Prefetch improves performance.
    .batch(BATCH_SIZE)           # Batch evaluation.
)

# ----------------------------------------------------------
# Now the data is ready for model building!
# Next step: Define a neural network classifier.
# ----------------------------------------------------------
