# -----------------------------------------------------------------------------
# High-Level Overview
# -----------------------------------------------------------------------------
# 1) Create synthetic noisy time series (sine waves + Gaussian noise).
# 2) Split data into training and validation sets.
# 3) Build a tf.data pipeline that converts the series into (window, next_value) pairs.
# 4) Define a simple DNN: Dense(10, ReLU) → Dense(10, ReLU) → Dense(1).
# 5) Perform a Learning Rate (LR) Finder: train for 100 epochs, exponentially
# increasing LR each epoch, record loss vs LR.
# 6) Plot loss vs LR, pick a stable LR (slightly below min-loss LR).
# 7) Rebuild and retrain the model with the chosen LR for 100 epochs.
# 8) Plot training loss vs epoch (full + zoomed after first few epochs).
# 9) Forecast on validation set, plot predictions vs actual, and compute MAE.
#
# -----------------------------------------------------------------------------
# Why Each Block Matters
# -----------------------------------------------------------------------------
# Synthetic Data:
# - Combines sine waves + noise to create predictable yet realistic series.
# - Seeds are fixed for reproducibility.
#
# Windowing:
# -----------------------------------------------------------------------------
# - Converts sequence into overlapping windows of length `window_size`.
# - Each window predicts the next value.
# - Implemented with tf.data for efficiency.
#
# DNN Architecture:
# - Simple dense network (10→10→1) for quick demonstration.
# - Not sequence-aware but works for short windows.
#
# Learning Rate Finder:
# - Starts with tiny LR, increases exponentially each epoch.
# - Plot loss vs LR → pick region where loss decreases before instability.
# - Helps avoid guesswork when selecting LR.
#
# Retraining:
# - Rebuild model with chosen LR.
# - Train for 100 epochs and monitor loss vs epoch.
# - Zoomed plot shows if learning continues beyond early epochs.
#
# Forecasting:
# - Slide windows over validation region.
# - Predict next values, align with validation set.
# - Evaluate with Mean Absolute Error (MAE) for interpretability.
#
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# LR Plot Interpretation
# -----------------------------------------------------------------------------
# - Very small LR → slow or flat loss.
# - Good LR region → rapid loss decrease.
# - Too large LR → unstable / exploding loss.
# - Pick LR slightly smaller than min-loss LR (e.g. 1/5 or 1/10).
#
# -----------------------------------------------------------------------------
# Useful Experiments
# -----------------------------------------------------------------------------
# - Change window_size for more/less context.
# - Add more layers/neurons or switch to sequence models (RNN, LSTM, GRU).
# - Try Adam optimizer for stability and faster convergence.
# - Add Dropout, BatchNorm, or Weight decay for regularization.
# - Tune batch size, epochs, or use EarlyStopping & ModelCheckpoint.
# - Normalize input windows for better convergence.
#
# -----------------------------------------------------------------------------
# Expected Results
# -----------------------------------------------------------------------------
# - LR plot shows a clear decreasing → unstable region.
# - Loss vs Epoch shows sharp fall then flattening.
# - Forecast follows main sine pattern, not noise.
# - MAE gives a numeric error measure.
# -----------------------------------------------



# dnn_windowed_time_series_with_lr_finder.py
# Fully commented, line-by-line example that builds a DNN to forecast the next value
# in a windowed time series, runs a learning-rate range test, picks a learning rate,
# retrains and plots results (loss vs lr, loss vs epoch, forecast vs actual, MAE).

import os                                                           # operating-system utilities
import random                                                       # python random for reproducibility
import numpy as np                                                   # numerical arrays and RNG
import tensorflow as tf                                              # tensorflow / keras
from tensorflow import keras                                         # keras shortcuts
from tensorflow.keras import layers                                  # common keras layers
import matplotlib.pyplot as plt                                      # plotting library

# -------------------------- reproducibility --------------------------
SEED = 42                                                             # a fixed random seed for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)                               # make python hashing deterministic
random.seed(SEED)                                                      # seed Python's random
np.random.seed(SEED)                                                   # seed NumPy's random
tf.random.set_seed(SEED)                                               # seed TensorFlow's random

# -------------------------- synthetic data ----------------------------
# create a simple, noisy periodic time series so we can train and inspect
time = np.arange(0.0, 200.0, 0.1)                                      # time steps from 0 to 200 with step 0.1
series = np.sin(0.1 * time)                                             # base sine wave
series += 0.2 * np.sin(0.5 * time)                                      # add a second harmonic for complexity
series += 0.1 * np.random.randn(len(time))                              # add Gaussian noise to make it realistic

# -------------------------- train/validation split --------------------
split_frac = 0.8                                                        # fraction of data for training
split_time = int(len(series) * split_frac)                              # absolute index to split on
train_series = series[:split_time]                                       # training portion of the series
valid_series = series[split_time:]                                       # validation portion we will forecast on

# -------------------------- windowing utility -------------------------
# create a tf.data pipeline that yields (window, next_value) pairs

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Turn a 1-D array `series` into a tf.data.Dataset of (input_window, label).

    Each input_window has length `window_size` and the label is the value that
    immediately follows that window in the original series.
    """
    series = tf.convert_to_tensor(series, dtype=tf.float32)             # convert numpy array into a TF tensor
    ds = tf.data.Dataset.from_tensor_slices(series)                      # build dataset of scalar time steps
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)        # create overlapping windows that include label
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))                 # convert each window into a tensor of shape (window_size+1,)
    ds = ds.shuffle(shuffle_buffer, seed=SEED)                           # shuffle window order to avoid sequence bias
    ds = ds.map(lambda w: (w[:-1], w[-1]))                               # split into (input_window, label)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)                # batch windows and prefetch for performance
    return ds

# -------------------------- hyperparameters ---------------------------
window_size = 20                                                        # how many past timesteps the model can see
batch_size = 32                                                         # examples per batch
shuffle_buffer = 1000                                                   # buffer size for shuffling windows

# prepare the training dataset using our utility function
train_ds = windowed_dataset(train_series, window_size, batch_size, shuffle_buffer)

# -------------------------- model builder -----------------------------
# wrap model creation in a function so we can recreate from scratch for retraining

def build_dnn(window_size):
    """Return a compiled Sequential DNN matching the description.

    Architecture: Dense(10, relu) -> Dense(10, relu) -> Dense(1, linear)
    Loss: MSE
    Metrics: Mean Absolute Error
    Optimizer: placeholder (set when compiling)
    """
    model = keras.models.Sequential()                                    # Sequential container for layers
    model.add(layers.Input(shape=(window_size,)))                         # input layer expects a vector of length `window_size`
    model.add(layers.Dense(10, activation='relu'))                        # first hidden layer with 10 neurons and ReLU activation
    model.add(layers.Dense(10, activation='relu'))                        # second hidden layer with 10 neurons and ReLU activation
    model.add(layers.Dense(1))                                            # output layer producing a single scalar prediction
    return model

# -------------------------- learning-rate range test ------------------
# we'll do a simple LR finder: increase lr exponentially each epoch and record loss

# build a fresh model for the LR test and compile with a tiny initial lr
lr_test_model = build_dnn(window_size)                                   # create a fresh model for the LR sweep
initial_lr = 1e-8                                                        # very small starting learning rate for the sweep
optimizer = keras.optimizers.SGD(learning_rate=initial_lr)               # use SGD as described in the video
lr_test_model.compile(loss='mse', optimizer=optimizer)                   # compile with MSE loss and the optimizer

# define a scheduler that increases the learning rate every epoch exponentially
def lr_scheduler(epoch):
    """Return a learning rate that increases exponentially with epoch.

    This follows the common formula: lr = initial_lr * 10^(epoch / factor).
    The factor controls how fast we increase (20 is a typical choice).
    """
    return initial_lr * (10 ** (epoch / 20.0))                            # compute lr for the current epoch

# Keras callback that updates the optimizer.lr every epoch using the function above
lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

# run the LR sweep for `epochs` epochs to see loss as a function of lr
lr_finder_epochs = 100                                                   # number of epochs for the LR range test (as in the video)
history_lr = lr_test_model.fit(train_ds, epochs=lr_finder_epochs, callbacks=[lr_callback], verbose=1)

# compute the list of learning rates used and corresponding losses
lrs = [lr_scheduler(e) for e in range(len(history_lr.history['loss']))] # learning rate per epoch
losses = history_lr.history['loss']                                      # loss per epoch recorded by Keras

# -------------------------- plot: loss vs learning rate ----------------
plt.figure(figsize=(8, 5))                                                # create a figure to display the LR curve
plt.semilogx(lrs, losses)                                                 # plot loss against lr on a log-x scale
plt.xlabel('Learning rate (log scale)')                                   # x-axis label
plt.ylabel('Loss (MSE)')                                                  # y-axis label
plt.title('Learning Rate Range Test: loss vs lr')                         # title
plt.grid(True, which='both', alpha=0.25)                                  # grid for easier reading
plt.show()                                                                 # show the plot

# -------------------------- pick a good learning rate ------------------
# simple automated heuristic: pick the lr that produced the minimum loss
min_loss_idx = int(np.argmin(losses))                                     # index of the smallest loss in the sweep
suggested_lr = lrs[min_loss_idx]                                           # learning rate corresponding to the smallest loss
print(f"[LR FINDER] minimum loss at epoch {min_loss_idx}, suggested lr = {suggested_lr:.2e}")

# heuristics note: often you choose a value slightly smaller than `suggested_lr`
# because the absolute minimum can be at the start of instability; you can
# multiply by 0.2 or 0.1 to be conservative. We'll pick a conservative lr here.
chosen_lr = suggested_lr * 0.2                                            # choose a slightly smaller lr for stable training
print(f"[LR CHOSEN] using chosen learning rate = {chosen_lr:.2e} for retraining")

# -------------------------- retrain with chosen lr ---------------------
# rebuild a fresh model so we start training from random initial weights
final_model = build_dnn(window_size)                                      # new model instance
final_optimizer = keras.optimizers.SGD(learning_rate=chosen_lr)           # SGD optimizer with our chosen lr
final_model.compile(loss='mse', optimizer=final_optimizer, metrics=[keras.metrics.MeanAbsoluteError()])

# train for a reasonable number of epochs and capture the history
retrain_epochs = 100                                                      # same number as in the description
history_final = final_model.fit(train_ds, epochs=retrain_epochs, verbose=1)

# -------------------------- plot: loss vs epoch ------------------------
plt.figure(figsize=(8,5))                                                 # new figure for loss-vs-epoch
plt.plot(history_final.history['loss'], label='loss')                     # plot all epochs
plt.xlabel('Epoch')                                                       # x-axis label
plt.ylabel('Loss (MSE)')                                                  # y-axis label
plt.title('Training Loss vs Epoch')                                       # title
plt.grid(alpha=0.25)                                                      # subtle grid
plt.legend()                                                              # show legend
plt.show()                                                                 # display plot

# show the later epochs more clearly by removing the first few (e.g. first 10)
start = 10                                                                 # index to start the zoom
plt.figure(figsize=(8,5))                                                 # figure for zoomed-in view
plt.plot(history_final.history['loss'][start:], label=f'loss (epochs {start}+)')
plt.xlabel('Epoch (offset)')                                               # x-axis label
plt.ylabel('Loss (MSE)')                                                   # y-axis label
plt.title('Training Loss (zoomed after early epochs)')                     # title
plt.grid(alpha=0.25)                                                       # grid
plt.legend()                                                               # legend
plt.show()                                                                  # display

# -------------------------- forecasting on validation ------------------
# helper that runs the model over sliding windows to produce a forecast sequence

def model_forecast(model, series, window_size):
    """Produce forecasts for each position where a full `window_size` exists.

    If you pass `series` that begins `window_size` steps before the validation
    start, the returned predictions will align with the validation labels.
    """
    series = tf.convert_to_tensor(series, dtype=tf.float32)               # ensure series is a tensor
    ds = tf.data.Dataset.from_tensor_slices(series)                        # dataset of scalars from the series
    ds = ds.window(window_size, shift=1, drop_remainder=True)              # windows of length `window_size`
    ds = ds.flat_map(lambda w: w.batch(window_size))                      # convert each window to a batch tensor
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)                          # efficient batching for prediction
    preds = model.predict(ds)                                              # run prediction over the dataset
    return preds.ravel()                                                    # flatten to 1-D array

# produce forecasts for the validation region by giving the model the slice
# that starts `window_size` steps before the validation split so first window
# contains the last `window_size` training points and predicts the first val point
forecast_input = series[split_time - window_size:]                         # series slice used to create windows that align with validation
forecast = model_forecast(final_model, forecast_input, window_size)         # predicted values aligned to valid_series indices

# ensure lengths match and compute MAE
forecast = forecast[:len(valid_series)]                                    # slice to exactly match validation length
mae = tf.keras.metrics.mean_absolute_error(valid_series, forecast).numpy() # compute mean absolute error
print(f"Validation MAE: {mae:.5f}")

# -------------------------- plot: forecast vs actual --------------------
plt.figure(figsize=(10,5))                                                  # create figure for forecast comparison
plt.plot(range(split_time, split_time + len(valid_series)), valid_series, label='Actual')  # plot actual validation values
plt.plot(range(split_time, split_time + len(valid_series)), forecast, label='Forecast')    # plot model forecast
plt.xlabel('Time index')                                                     # x-axis label
plt.ylabel('Value')                                                          # y-axis label
plt.title('Forecast vs Actual on Validation Set')                            # title
plt.legend()                                                                 # show legend
plt.grid(alpha=0.25)                                                         # grid
plt.show()                                                                    # display

# -------------------------- wrap-up message ----------------------------
print('\nFinished.\n')
print('Experiment: try changing the window_size, layer sizes, or optimizer.')
print('Try adding Dropout, BatchNormalization, or using Adam instead of SGD.')
print('You can re-run the LR sweep with different lr ranges or epoch counts.')



# Running the Code

# - Outputs:
# * Loss vs LR plot.
# * Loss vs Epoch plot (full + zoomed).
# * Forecast vs Actual plot.
# * Printed LR choice + MAE.
