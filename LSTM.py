# ================================================================
# Summary of Key Points (CNN, LSTM, and Overfitting in Text Classification)
# ================================================================

# 1. LSTMs (Long Short-Term Memory) excel at capturing long-range dependencies 
#    in text, making them suitable for sequential patterns and contextual meaning.

# 2. CNNs (Convolutional Neural Networks) capture local word patterns (n-grams) 
#    using filters, and are generally faster and parallelizable compared to LSTMs.

# 3. In CNNs, filters (e.g., size 5) slide across sequences, 
#    detecting meaningful phrases like “not good at all.”

# 4. LSTMs process input word-by-word, storing information in hidden states and gates, 
#    while CNNs detect features in chunks of words.

# 5. CNNs are computationally lighter and faster, while LSTMs are heavier but 
#    better at modeling long-term dependencies.

# 6. CNNs can easily overfit if too many filters or layers are used 
#    without proper regularization techniques.

# 7. Overfitting is a bigger challenge in text than in images 
#    because of Out-of-Vocabulary (OOV) words in validation/test datasets.

# 8. OOV words are words that never appeared in training, 
#    so the model cannot classify them properly.

# 9. More OOV words = weaker generalization and higher risk of overfitting 
#    (training accuracy very high, validation accuracy much lower).

# 10. With CNNs, we often observe training accuracy near 100% 
#     but validation accuracy stagnates around ~70–80%, indicating overfitting.

# 11. LSTMs may generalize slightly better than CNNs, 
#     but they also suffer from OOV and limited training data issues.

# 12. Regularization (dropout, early stopping, weight decay, smaller models) 
#     is essential to reduce overfitting in both CNNs and LSTMs.

# 13. Using pretrained embeddings (GloVe, Word2Vec, BERT) 
#     helps mitigate OOV issues by providing better word representations.

# 14. Data augmentation techniques (synonym replacement, back-translation, paraphrasing) 
#     can reduce overfitting by introducing more text variety.

# 15. Both CNNs and LSTMs are powerful for text classification:
#     - CNNs are faster and focus on local context
#     - LSTMs are slower but capture long-term sequential context
#     - Both need careful tuning to avoid overfitting.




# ==========================================================
# Text Classification with LSTM and Bidirectional LSTM
# ==========================================================
# In this example, we demonstrate how to classify text (sentiment analysis)
# using an LSTM (Long Short-Term Memory) network. We’ll also use Bidirectional
# LSTM, which allows the model to capture context from both past and future words.
# 
# Concepts Covered:
# 1) Why LSTMs are better than simple RNNs
# 2) How embeddings turn words into dense vectors
# 3) How Bidirectional LSTM doubles the context
# 4) How dense layers make final predictions
# ==========================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# ----------------------------------------------------------
# 1) Example dataset (toy data for demonstration)
# ----------------------------------------------------------
# Here we define a small dataset of movie reviews with labels:
#   - 1 means positive sentiment
#   - 0 means negative sentiment
# In real-world cases, this would be a large dataset (like IMDB reviews).
sentences = [
    "I love this movie",
    "This film was terrible",
    "I really enjoyed the story",
    "The acting was bad",
    "An excellent film",
    "I hated the ending"
]
labels = [1, 0, 1, 0, 1, 0]   # Sentiment labels

# ----------------------------------------------------------
# 2) Tokenization and sequence preparation
# ----------------------------------------------------------
# Neural networks work with numbers, not raw text.
# Tokenization = breaking text into integers (each word = an index).
# We also handle Out-Of-Vocabulary (OOV) words with a special token <OOV>.
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Convert sentences into sequences of integers.
# Example: "I love this movie" → [2, 5, 7, 9] (numbers vary depending on vocab).
sequences = tokenizer.texts_to_sequences(sentences)

# Sequences have different lengths (some sentences are short, some are long).
# Neural networks require fixed-length inputs → we use padding.
# Padding adds zeros at the end so that all sequences are of equal length.
padded = pad_sequences(sequences, padding='post')

# Convert labels to numpy array for training.
labels = np.array(labels)

# ----------------------------------------------------------
# 3) Model definition
# ----------------------------------------------------------
# Now we define our neural network model using Keras Sequential API.
model = Sequential()

# Embedding Layer:
# - Words are represented as dense vectors instead of one-hot vectors.
# - Input_dim = size of vocabulary (1000 words max).
# - Output_dim = size of embedding vector for each word (here 64).
# - Input_length = length of each padded sequence.
# Embeddings help the model learn word similarities (e.g., "good" and "great"
# should have similar representations).
model.add(Embedding(input_dim=1000, output_dim=64, input_length=padded.shape[1]))

# Bidirectional LSTM Layer:
# - LSTM = Long Short-Term Memory, a special RNN that handles long-term context
#   using cell states and gates (forget, input, output gates).
# - Units = 64 → the dimensionality of the output space.
# - Bidirectional wrapper → runs the LSTM forward and backward.
#   This means the model can use both past (left context) and future (right context).
# - Output size = 64 * 2 = 128, because forward and backward outputs are concatenated.
model.add(Bidirectional(LSTM(64)))

# Dense hidden layer:
# - Fully connected layer with ReLU activation for non-linearity.
# - This layer learns higher-level patterns from the LSTM output.
model.add(Dense(64, activation='relu'))

# Output layer:
# - Single neuron with Sigmoid activation.
# - Sigmoid outputs a probability between 0 and 1.
# - If output > 0.5 → positive sentiment, else negative.
model.add(Dense(1, activation='sigmoid'))

# ----------------------------------------------------------
# 4) Model Compilation
# ----------------------------------------------------------
# - Loss function = binary_crossentropy (since it’s binary classification).
# - Optimizer = Adam (adaptive learning rate optimizer, very effective in practice).
# - Metrics = accuracy (to measure how many predictions are correct).
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ----------------------------------------------------------
# 5) Train the Model
# ----------------------------------------------------------
# model.summary() shows the layers, parameters, and output shapes.
# model.fit() trains the model:
#   - Inputs = padded sequences
#   - Outputs = sentiment labels
#   - Epochs = how many times the model sees the full dataset.
model.summary()
model.fit(padded, labels, epochs=10, verbose=1)

# ----------------------------------------------------------
# 6) Make a Prediction
# ----------------------------------------------------------
# Let's test our model on a new sentence.
# Step 1: Convert the sentence into a sequence.
test_sentence = ["I think the film was amazing"]
test_seq = tokenizer.texts_to_sequences(test_sentence)

# Step 2: Pad the sequence so it matches the training input size.
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1], padding='post')

# Step 3: Predict sentiment (closer to 1 = positive, closer to 0 = negative).
prediction = model.predict(test_pad)

print("Prediction (closer to 1 = positive sentiment):", prediction)

# ==========================================================
# 📌 THEORY SUMMARY
# - Problem: Simple RNNs forget long-term dependencies → can’t connect "Ireland"
#   to "Gaelic" if they’re far apart.
# - Solution: LSTMs use memory cells and gates to keep/forget information.
# - Bidirectional LSTMs read sequences both ways → better context.
# - Embeddings learn word meanings → similar words have similar vectors.
# - Dense layers classify based on features extracted by LSTMs.
# ==========================================================






# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================
# ==========================================================




# ==========================================================
# Text Classification with 1D Convolutional Neural Networks (CNNs)
# ==========================================================
# In this example, instead of LSTMs, we use CNNs (Conv1D) for text classification.
# CNNs are commonly used in image processing, but they also work well for text.
#
# ESSENCE OF CNN FOR TEXT:
# 1) Each word embedding is treated like a "pixel" in a 1D sequence.
# 2) Convolution filters slide over fixed-size word windows (e.g., 5 words).
# 3) Each filter learns to detect useful n-gram patterns like "not good at all".
# 4) Multiple filters = multiple features extracted from text.
# 5) CNNs are computationally efficient and parallelizable.
# 6) They capture local dependencies well, but may miss very long-range context.
#
# COMPARISON TO LSTM:
# - LSTM captures long-term dependencies with memory gates.
# - CNN captures local n-gram-like features with sliding filters.
# - LSTM is slower but better for long context, CNN is faster but local.
# - CNN can overfit quickly without proper regularization.
# ==========================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# ----------------------------------------------------------
# 1) Example dataset (toy dataset for demonstration)
# ----------------------------------------------------------
sentences = [
    "I love this movie",
    "This film was terrible",
    "I really enjoyed the story",
    "The acting was bad",
    "An excellent film",
    "I hated the ending"
]
labels = [1, 0, 1, 0, 1, 0]   # 1 = positive, 0 = negative sentiment

# ----------------------------------------------------------
# 2) Tokenization and sequence preparation
# ----------------------------------------------------------
# Convert text to sequences of integers (word indexes).
# num_words = vocabulary size (max words we want to consider).
# oov_token = token for words not seen during training (<OOV>).
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Convert sentences into integer sequences.
sequences = tokenizer.texts_to_sequences(sentences)

# Pad sequences so that all have the same length (important for CNN input).
padded = pad_sequences(sequences, padding='post')

# Convert labels into numpy array for training.
labels = np.array(labels)

# ----------------------------------------------------------
# 3) Model definition (CNN architecture)
# ----------------------------------------------------------
model = Sequential()

# Embedding Layer:
# - Converts integer word indices into dense vectors.
# - input_dim = vocabulary size (1000 words max).
# - output_dim = embedding size (vector size for each word, here 64).
# - input_length = sequence length after padding.
model.add(Embedding(input_dim=1000, output_dim=64, input_length=padded.shape[1]))

# Convolution Layer (1D):
# - filters = number of filters (128) → number of feature maps learned.
# - kernel_size = 5 → each filter spans 5 consecutive words.
# - activation = relu → introduces non-linearity.
# This layer scans through the sentence and learns local patterns (e.g., phrases).
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))

# Global Max Pooling Layer:
# - Takes the maximum value from each feature map (128 in this case).
# - Reduces data from (sequence_length, filters) → (filters).
# - Captures the most important feature detected by each filter.
model.add(GlobalMaxPooling1D())

# Dense hidden layer:
# - Fully connected layer with ReLU activation.
# - Learns interactions between the features extracted by CNN.
model.add(Dense(64, activation='relu'))

# Output layer:
# - Single neuron with Sigmoid activation.
# - Outputs probability of positive sentiment (1) vs negative sentiment (0).
model.add(Dense(1, activation='sigmoid'))

# ----------------------------------------------------------
# 4) Model Compilation
# ----------------------------------------------------------
# - Loss function: binary_crossentropy (since it’s binary classification).
# - Optimizer: Adam (adaptive learning rate optimizer).
# - Metrics: accuracy (how many predictions are correct).
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# ----------------------------------------------------------
# 5) Train the Model
# ----------------------------------------------------------
# model.summary() shows the structure of the CNN.
# model.fit() trains the model on the dataset.
model.summary()
model.fit(padded, labels, epochs=10, verbose=1)

# ----------------------------------------------------------
# 6) Make a Prediction
# ----------------------------------------------------------
# Test the trained model on a new sentence.
test_sentence = ["I think the film was amazing"]
test_seq = tokenizer.texts_to_sequences(test_sentence)

# Pad the sequence to match input size.
test_pad = pad_sequences(test_seq, maxlen=padded.shape[1], padding='post')

# Get prediction (closer to 1 = positive, closer to 0 = negative).
prediction = model.predict(test_pad)

print("Prediction (closer to 1 = positive sentiment):", prediction)

# ==========================================================
# 📌 THEORY SUMMARY
# - CNNs for text treat word embeddings as 1D signals.
# - Filters (kernels) detect n-gram patterns across sequences.
# - Pooling layers reduce dimensions and capture key features.
# - CNNs are faster and parallelizable compared to LSTMs.
# - They excel at capturing local dependencies but may miss long-term ones.
# - Overfitting is a common risk → regularization is needed in real cases.
# ==========================================================













