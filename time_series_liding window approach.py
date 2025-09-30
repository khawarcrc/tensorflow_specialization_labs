# --------------------------------------------------------
# STEP 1: Import libraries
# --------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# --------------------------------------------------------
# STEP 2: Create a synthetic time series
# --------------------------------------------------------
# We'll make a simple sine wave with added noise.
# This simulates a real-world time series that has patterns + randomness.
time = np.arange(0, 100, 0.1)                  # Time steps from 0 to 100 with step 0.1
series = np.sin(time) + 0.1 * np.random.randn(len(time))  

# --------------------------------------------------------
# STEP 3: Visualize the series
# --------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(time, series, label="Time Series")
plt.title("Synthetic Time Series (Sine + Noise)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()


# --------------------------------------------------------
# STEP 4: Define the windowed_dataset function
# --------------------------------------------------------
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """
    Prepares a time series dataset for supervised learning using a sliding window approach.
    
    Args:
        series (array-like): The input time series data (e.g., a NumPy array or list).
        window_size (int): Number of previous time steps to use as input features (X).
        batch_size (int): Number of samples per training batch.
        shuffle_buffer (int): Buffer size for shuffling (controls randomness vs speed).
    
    Returns:
        tf.data.Dataset: A dataset of (features, labels) for training.
    """
    # Step 1: Turn the series into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Step 2: Create sliding windows of size (window_size + 1)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Step 3: Flatten each window into a tensor
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Step 4: Shuffle the windows for randomness
    dataset = dataset.shuffle(shuffle_buffer)

    # Step 5: Split into (X, y) → X = all values except last, y = last value
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Step 6: Batch and prefetch for training efficiency
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset



# --------------------------------------------------------
# STEP 5: Prepare the dataset
# --------------------------------------------------------
window_size = 20        # Number of previous steps used as features
batch_size = 32         # Training batch size
shuffle_buffer = 1000   # Buffer size for shuffling

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer)

# --------------------------------------------------------
# STEP 6: Inspect a few (X, y) pairs
# --------------------------------------------------------
for X, y in dataset.take(1):   # Just take one batch
    print("X shape:", X.shape) # (batch_size, window_size)
    print("y shape:", y.shape) # (batch_size,)
    print("First example X:", X[0].numpy())
    print("First example y:", y[0].numpy())


# --------------------------------------------------------
# STEP 7: Build a simple neural network
# --------------------------------------------------------
# We'll use a small Sequential model:
# - Dense(64) with ReLU activation
# - Dense(1) for output (predicting next value)
model = keras.models.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=[window_size]),
    keras.layers.Dense(1)
])

# Compile model with mean squared error loss (good for regression)
model.compile(
    loss="mse",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mae"]  # mean absolute error for interpretability
)

model.summary()


# --------------------------------------------------------
# STEP 8: Train the model
# --------------------------------------------------------
history = model.fit(dataset, epochs=10)

# Plot training loss
plt.plot(history.history["loss"], label="Training Loss (MSE)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.show()


# --------------------------------------------------------
# STEP 9: Make predictions
# --------------------------------------------------------
# We'll predict values step by step using the trained model.

# Start with the first window
forecast = []
window = series[:window_size]  # Initial window

for i in range(len(series) - window_size):
    # Reshape to match input (1, window_size)
    x_input = np.expand_dims(window, axis=0)
    
    # Predict the next value
    prediction = model.predict(x_input, verbose=0)
    
    # Save prediction
    forecast.append(prediction[0, 0])
    
    # Slide the window forward by appending the true next value
    window = np.append(window[1:], series[i + window_size])

forecast = np.array(forecast)


# --------------------------------------------------------
# STEP 10: Visualize predictions vs actual values
# --------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(time[window_size:], series[window_size:], label="Actual Series")
plt.plot(time[window_size:], forecast, label="Predictions")
plt.title("Predicted vs Actual Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
