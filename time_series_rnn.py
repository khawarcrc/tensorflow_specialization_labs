# ============================================================
# 🧠 PROBLEM STATEMENT
# ============================================================
# We are tasked with building a Recurrent Neural Network (RNN)
# that can predict future values in a time series.
#
# A time series is a sequence of data points measured over time,
# such as temperature, stock prices, or sales data.
# Our goal is to train a model that learns from the past values
# to forecast future values — for example:
#
#     [x₁, x₂, x₃, ..., xₜ]  → predict xₜ₊₁
#
# To achieve this, we’ll:
#   1️⃣ Generate a synthetic noisy sinusoidal time series.
#   2️⃣ Convert it into a supervised learning dataset
#       (by creating windows of past values to predict the next one).
#   3️⃣ Build and train an RNN model with two stacked SimpleRNN layers.
#   4️⃣ Use a "Learning Rate Finder" callback to identify the best learning rate.
#   5️⃣ Use the Huber loss function (robust to noise/outliers).
#   6️⃣ Retrain the model using the optimal learning rate.
#   7️⃣ Evaluate its Mean Absolute Error (MAE) on validation data.
#   8️⃣ Visualize actual vs predicted series.
#
# This is a foundational exercise in sequence modeling using RNNs.
# ============================================================

# ============================================================
# 🧩 STEPS TO SOLVE THE PROBLEM
# ============================================================
# Step 1: Import necessary libraries (TensorFlow, NumPy, Matplotlib).
# Step 2: Generate a synthetic time series dataset with noise.
# Step 3: Split data into training and validation sets.
# Step 4: Create a windowed dataset generator to prepare data for the RNN.
# Step 5: Build an RNN model with two SimpleRNN layers.
# Step 6: Implement a learning rate finder to identify the best LR.
# Step 7: Retrain the model using the optimal LR.
# Step 8: Forecast future values using the trained model.
# Step 9: Evaluate the model using MAE and visualize results.
# ============================================================


# ------------------------------------------------------------
# Step 1: Import required libraries
# ------------------------------------------------------------
import tensorflow as tf      # TensorFlow provides RNN layers, optimizers, etc.
import numpy as np           # NumPy for numerical operations and data handling
import matplotlib.pyplot as plt  # Matplotlib for plotting results


# ------------------------------------------------------------
# Step 2: Create a synthetic time series
# ------------------------------------------------------------

# Set a random seed for reproducibility — ensures same random numbers every run
np.random.seed(42)

# Create a sequence of 4 years of daily data points (365 days * 4 + 1 = 1461)
time = np.arange(4 * 365 + 1, dtype="float32")

# Define the series:
# - Base pattern: a sine wave to simulate periodic behavior.
# - Add Gaussian noise to make the data more realistic (like real-world signals).
series = np.sin(time * 0.1) + np.random.normal(scale=0.1, size=len(time))

# ------------------------------------------------------------
# Step 3: Split dataset into training and validation portions
# ------------------------------------------------------------

# We use the first 1000 points for training and the rest for validation.
split_time = 1000
time_train = time[:split_time]       # Training timestamps
x_train = series[:split_time]        # Training values
time_valid = time[split_time:]       # Validation timestamps
x_valid = series[split_time:]        # Validation values


# ------------------------------------------------------------
# Step 4: Create windowed dataset function
# ------------------------------------------------------------

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """
    Converts a 1D time series into a dataset of (input_window, label) pairs.

    For example, if the series = [1, 2, 3, 4, 5] and window_size = 3,
    we create samples like:
        Input = [1, 2, 3] → Label = 4
        Input = [2, 3, 4] → Label = 5

    Additionally, the input is reshaped into (window_size, 1)
    because RNNs expect 3D inputs: (batch_size, time_steps, features)
    """
    # Convert the series into a TensorFlow dataset object
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Create sliding windows of size (window_size + 1)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten nested windows into simple tensors
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
    
    # Shuffle the dataset to ensure the model doesn’t see ordered data
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Split each window into (input_window, label)
    dataset = dataset.map(lambda w: (w[:-1], w[-1]))
    
    # Expand input dimensions to match RNN expected input (time_steps, features)
    dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=-1), y))
    
    # Batch the data and prefetch for performance
    return dataset.batch(batch_size).prefetch(1)


# Define parameters for windowing
window_size = 30          # Each input will contain 30 consecutive time steps
batch_size = 32           # Number of samples processed before updating weights
shuffle_buffer_size = 1000  # Shuffle buffer for randomness

# Create the actual training dataset from the time series
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


# ------------------------------------------------------------
# Step 5: Build the RNN model
# ------------------------------------------------------------

model = tf.keras.models.Sequential([
    # Input layer specifying that each sample has shape (window_size, 1)
    tf.keras.Input(shape=(window_size, 1)),
    
    # First RNN layer with 40 neurons
    # 'return_sequences=True' → outputs the hidden state for each time step
    tf.keras.layers.SimpleRNN(40, return_sequences=True),
    
    # Second RNN layer with 40 neurons
    # This one outputs only the final state (suitable for sequence-to-vector tasks)
    tf.keras.layers.SimpleRNN(40),
    
    # Dense output layer with 1 neuron (predicts a single next value)
    tf.keras.layers.Dense(1),
    
    # Lambda layer scales up the output (useful if values are small)
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])


# ------------------------------------------------------------
# Step 6: Learning Rate Finder Setup
# ------------------------------------------------------------

# Define the loss function
# Huber loss = combines MSE (for small errors) and MAE (for large errors)
# It’s robust against outliers and noise in the data.
loss_fn = tf.keras.losses.Huber()

# Define optimizer
# Using Stochastic Gradient Descent (SGD) with momentum
# Start with a very small learning rate for the learning rate finder
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9)

# Compile model with loss function and optimizer
model.compile(loss=loss_fn, optimizer=optimizer)

# Define Learning Rate Scheduler callback
# Gradually increases LR exponentially each epoch → helps visualize best LR range
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20)
)


# ------------------------------------------------------------
# Step 7: Run Learning Rate Finder
# ------------------------------------------------------------

# Train for 100 epochs while increasing learning rate
# The goal is NOT to get good accuracy here — just to observe the loss curve.
history = model.fit(train_set, epochs=100, callbacks=[lr_scheduler])

# Create a list of all learning rates used
lrs = 1e-8 * (10 ** (np.arange(100) / 20))

# Plot loss vs learning rate to identify the optimal range
plt.figure(figsize=(10, 6))
plt.semilogx(lrs, history.history["loss"])  # Log scale for LR axis
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate Finder")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Step 8: Retrain Model with Tuned Learning Rate
# ------------------------------------------------------------

# From the plot, suppose we find that best LR ≈ 1e-5
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)

# Re-compile with new optimizer and include Mean Absolute Error (MAE) as a metric
model.compile(loss=loss_fn, optimizer=optimizer, metrics=["mae"])

# Train model for 400 epochs — enough to converge without overfitting
history = model.fit(train_set, epochs=400)


# ------------------------------------------------------------
# Step 9: Forecasting Helper Function
# ------------------------------------------------------------

def model_forecast(model, series, window_size):
    """
    Generates predictions across a time series using a sliding window.

    The idea:
    - Take a moving window of 'window_size' previous points.
    - Predict the next one.
    - Move forward by one step and repeat.
    """
    # Convert series to dataset
    ds = tf.data.Dataset.from_tensor_slices(series)
    
    # Create sliding windows
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    
    # Flatten each window into a tensor
    ds = ds.flat_map(lambda w: w.batch(window_size))
    
    # Add channel dimension (feature dimension = 1)
    ds = ds.map(lambda w: tf.expand_dims(w, axis=-1))
    
    # Batch and prefetch for faster predictions
    ds = ds.batch(32).prefetch(1)
    
    # Predict all windows
    forecast = model.predict(ds)
    
    return forecast


# Use forecasting function to generate predictions for validation portion
rnn_forecast = model_forecast(model, series, window_size)

# Extract the relevant predictions matching validation time range
rnn_forecast = rnn_forecast[split_time - window_size:-1, 0]


# ------------------------------------------------------------
# Step 10: Evaluate and Visualize Results
# ------------------------------------------------------------

# Compute Mean Absolute Error (MAE) between predictions and actual validation values
mae = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()

# Print MAE to evaluate performance
print(f"Validation MAE: {mae:.3f}")

# Plot actual vs predicted time series for visual comparison
plt.figure(figsize=(10, 6))
plt.plot(time_valid, x_valid, label="Actual Series")       # True data
plt.plot(time_valid, rnn_forecast, label="RNN Forecast")   # Model predictions
plt.title("RNN Time Series Forecast (2-layer SimpleRNN)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
