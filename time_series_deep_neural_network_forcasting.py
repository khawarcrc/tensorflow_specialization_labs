# ============================================================
# 🔑 Key Concepts Explained
# ============================================================

# Windowing
# → Converts sequential time-series data into (features, label) pairs 
#   so that a neural network can learn temporal dependencies. 
# → Each window contains 'window_size' values as input and the next value as label. 
# → This transforms the problem into supervised learning where features → label. 
# → Critical for time-series forecasting because raw sequential data is not directly usable by DNNs.

# Dense Layers
# → A Dense layer performs: Output = Activation(W * Input + Bias).
# → ReLU activation (Rectified Linear Unit) introduces nonlinearity,
#   helping the model learn complex patterns beyond simple straight lines.
# → Multiple Dense layers stacked together form a Deep Neural Network (DNN).

# Output Layer
# → A single neuron with no activation (linear output) is used
#   because we are predicting a continuous numerical value (regression).
# → In time-series forecasting, this neuron predicts the next point in the sequence.

# MSE Loss (Mean Squared Error)
# → MSE = (1/N) * Σ (y_true - y_pred)^2
# → Squaring penalizes larger mistakes more heavily, which stabilizes regression training.
# → It provides a smooth gradient, making optimization easier for gradient descent.

# SGD Optimizer (Stochastic Gradient Descent)
# → Iteratively updates model weights based on error gradients. 
# → Momentum term is added to smooth updates and escape small local minima. 
# → Good balance between efficiency and stability, especially with large datasets.

# Learning Rate Scheduler
# → Dynamically adjusts learning rate during training. 
# → Instead of fixing LR, we test across a range of values to find the optimal one. 
# → Helps balance: small LR = slow convergence, large LR = unstable training. 
# → This is implemented using TensorFlow Callbacks.

# Log-scale Loss Curve
# → A diagnostic visualization: plots Loss vs Learning Rate on a logarithmic scale. 
# → Allows us to see the stability and efficiency of learning at different LRs. 
# → The "optimal LR region" is usually just before the loss starts increasing again.

# Retraining with Best LR
# → Once the best learning rate is identified, we fix it and retrain the network for longer epochs. 
# → This speeds up convergence and improves accuracy compared to arbitrary LR choices. 
# → Ensures the model avoids underfitting (too slow) or exploding gradients (too fast).

# Plotting Loss Curves
# → Visualization of model learning progress over epochs. 
# → Helps diagnose underfitting (loss remains high), overfitting (validation diverges), 
#   or healthy convergence (loss decreases smoothly). 
# → Cropping noisy early epochs often clarifies long-term training behavior.

# ============================================================
# Additional Important Concepts in This Lab
# ============================================================

# Batch Size
# → Number of samples processed before model updates weights. 
# → Smaller batch = more updates (faster learning, noisier gradients). 
# → Larger batch = smoother gradients, but may require more memory.

# Shuffling
# → Randomizes the order of training data to prevent model from learning spurious sequence patterns. 
# → Essential in windowed datasets where consecutive windows might be too similar.

# Prefetching
# → Loads the next batch of data while the current one is being processed by the model. 
# → Improves training performance by overlapping computation and data loading.

# Epochs
# → One full pass over the entire training dataset. 
# → More epochs = more opportunities to learn, but risk of overfitting if too many. 
# → Loss curves are used to decide when to stop training.

# Callbacks
# → Special TensorFlow functions that run at specific points during training 
#   (e.g., end of each epoch). 
# → Used here to implement Learning Rate Scheduling. 
# → Can also be used for Early Stopping, Checkpointing, or Logging.

# Neural Network Depth
# → Refers to the number of hidden layers. 
# → Deeper networks capture more complex representations. 
# → But too deep without enough data can cause overfitting or vanishing gradients.

# Momentum in Optimization
# → Concept borrowed from physics: accelerates SGD updates in the right direction. 
# → Prevents oscillations in steep valleys of the loss landscape. 
# → Makes training more stable and faster.

# Generalization
# → The model's ability to perform well on unseen data. 
# → The ultimate goal in machine learning: not just memorizing training data,
#   but learning patterns that apply to future predictions.

# ============================================================







# ==============================================================
# LAB: Deep Neural Network for Time Series Forecasting
# ==============================================================

# --------------------------------------------------------------
# STEP 0: Import necessary libraries
# --------------------------------------------------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow -> used for deep learning (building and training models)
# NumPy -> for numerical computations
# Matplotlib -> for visualizing results such as loss curves


# --------------------------------------------------------------
# STEP 1: Windowing the dataset
# --------------------------------------------------------------
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """
    Convert a time series into supervised learning data.
    
    Concept: 
    - Windowing → Converts sequential data into supervised learning format.
    - For each 'window' of values, we use the first N values as input (features)
      and the next value as the output (label).
    - This allows the neural network to learn temporal patterns.
    """
    dataset = tf.data.Dataset.from_tensor_slices(series)          # Convert series into a TensorFlow dataset
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)  # Create windows of size window_size+1
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1)) # Flatten windows into tensors
    dataset = dataset.shuffle(shuffle_buffer)                     # Shuffle for randomness (prevents overfitting to sequence order)
    dataset = dataset.map(lambda window: (window[:-1], window[-1])) # Split each window -> (features, label)
    dataset = dataset.batch(batch_size).prefetch(1)               # Batch & prefetch for performance
    return dataset


# --------------------------------------------------------------
# STEP 2: Create training dataset
# --------------------------------------------------------------
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


# --------------------------------------------------------------
# STEP 3: Define the Deep Neural Network (DNN) model
# --------------------------------------------------------------
"""
Concept: Dense Layers
- Each Dense layer applies a linear transformation (weights * inputs + bias)
  followed by an activation function.
- Activation = ReLU → introduces nonlinearity, allowing the network to learn
  more complex relationships than simple linear regression.

Output Layer
- 1 neuron → because we are predicting a single future value in the time series.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]), # Hidden layer 1
    tf.keras.layers.Dense(10, activation="relu"),                            # Hidden layer 2
    tf.keras.layers.Dense(1)                                                 # Output layer
])


# --------------------------------------------------------------
# STEP 4: Compile the model
# --------------------------------------------------------------
"""
Concept: Loss Function
- MSE (Mean Squared Error) → Penalizes squared differences between predicted
  and actual values.
- Works well for regression problems such as time series forecasting.

Concept: Optimizer
- SGD (Stochastic Gradient Descent) → Updates weights step by step.
- Momentum term → helps accelerate updates in the right direction and avoids
  getting stuck in small local minima.
"""

model.compile(loss="mse",
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9))


# --------------------------------------------------------------
# STEP 5: Learning Rate Scheduler Callback
# --------------------------------------------------------------
"""
Concept: Learning Rate Scheduler
- Instead of picking a fixed learning rate, we let it grow each epoch.
- Formula: 1e-8 * 10^(epoch/20)
- This way, we can observe how the model performs at different learning rates.
- Helps us find the "sweet spot" where loss decreases the most effectively.
"""

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20)
)

history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])


# --------------------------------------------------------------
# STEP 6: Plot Loss vs Learning Rate
# --------------------------------------------------------------
"""
Concept: Log-scale Plot
- x-axis = Learning rate (logarithmic scale)
- y-axis = Loss
- We pick the region where loss is lowest and stable.
- For example, if minimum loss is near 7e-6, that's our optimal learning rate.
"""

lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])   # semilogx = log scale on x-axis
plt.axis([1e-8, 1e-3, 0, max(history.history["loss"])])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Loss vs Learning Rate")
plt.show()


# --------------------------------------------------------------
# STEP 7: Retrain Model with Best Learning Rate
# --------------------------------------------------------------
"""
Concept: Retraining with Best LR
- Once optimal LR is chosen, retrain the network for more epochs.
- This improves convergence speed and final accuracy.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse",
              optimizer=tf.keras.optimizers.SGD(learning_rate=7e-6, momentum=0.9))

history = model.fit(dataset, epochs=500)


# --------------------------------------------------------------
# STEP 8: Plot Training Loss Over Time
# --------------------------------------------------------------
"""
Concept: Plotting Loss Curves
- Loss curve helps us diagnose if the model is learning or overfitting.
- Early epochs often have very high loss → skewing visualization.
- Cropping them off (after 10 epochs) gives a clearer view.
"""

plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Cropped view (ignore first 10 epochs)
plt.plot(history.history["loss"][10:])
plt.xlabel("Epochs (after 10)")
plt.ylabel("Loss")
plt.title("Training Loss (Epoch > 10)")
plt.show()
