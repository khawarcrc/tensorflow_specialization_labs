import tensorflow as tf
import numpy as np  


model = tf.keras.model.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]))
])

# Sequential: Defines a stack of layers (a simple linear model in this case).
# Dense(units=1): A single neuron (simplest neural network).
# input_shape=[1]: One input value per example (just a number X).

model.compile(optimizer='sgd', loss='mean_squared_error')

# loss='mean_squared_error': Calculates the difference between predicted Y and actual Y.
# optimizer='sgd': Stochastic Gradient Descent — adjusts the model to reduce the loss.

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=float)

model.fit(xs,ys, epoch=500)

# Step	Concept/Action
# Setup	Import TensorFlow and NumPy
# Define Model	A simple neural network with 1 layer and 1 neuron
# Compile Model	Use MSE loss and SGD optimizer
# Provide Data	Input/output pairs (X and Y) using NumPy
# Train Model	500 epochs of guessing, measuring, and improving
# Monitor Loss	Observe training progress by watching the loss get smaller
# Predict New Data	Test with unseen input and interpret the predicted result

# Term	Meaning
# Epoch	One complete pass through the training process
# Loss	A measurement of how wrong the model's prediction is
# Optimizer	A method to adjust the model based on the loss
# Training	The process of learning from known data
# Prediction	Applying the learned pattern to new/unseen data
# Generalization	The model's ability to handle new, unseen inputs


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),   # Flattened into 784 values
    tf.keras.layers.Dense(128, activation='relu'),   # Passed through 128 neurons to extract complex features
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons for classification    
])

# tf.keras.layers.Flatten(input_shape=(28, 28)),
# Images are 28×28 pixels → a 2D grid.
# Flatten converts the 2D array into a 1D array of 784 values.
# These 784 values (pixels) become inputs to the next layer.
# You can think of this as unrolling the image into a list of pixel values.

# tf.keras.layers.Dense(128, activation='relu'),
# This is the "interesting" layer — also called the hidden layer.
# Each of the 784 inputs is connected to 128 neurons.
# The relu activation function introduces non-linearity and helps the model learn complex patterns.
# Think of this as computing something like:
# y = w1*x1 + w2*x2 + w3*x3 + ... + w128*x128

# tf.keras.layers.Dense(10, activation='softmax')
# Final layer contains 10 neurons, one for each clothing category.
# The softmax activation outputs probabilities for each class.


class MyCallback(tf.keras.callbacks.Callback):   # Custom callback to stop training early
    def on_epoch_end(self, epoch, logs={}):      # on_epoch_end is called at the end of each epoch
        if logs.get('loss') < 0.4: 
            print("\nReached 60% accuracy so cancelling training!") 
            self.model.stop_training = True
            

# CNN model  
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1)), #  Convolutional layer with 64 filters of size 3x3
    tf.keras.layers.MaxPooling2D(2,2),  # Max pooling layer to reduce dimensionality
    tf.keras.layers.Flatten(),  # Flatten layer to convert 2D output into 1D for the dense layer
    tf.keras.layers.Dense(128, activation='relu'),   # Dense (fully connected) layer with 128 neurons and relu activation 
    tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class) and softmax for probability distribution
])


# Layer	Purpose
# Conv2D	Scans the image with 64 filters of size 3×3 to detect local patterns
# MaxPooling2D	Reduces dimensionality (and computation) by taking the max in 2×2 regions
# Flatten	Flattens the 2D output into a 1D vector for the dense layer
# Dense (128)	Fully connected layer to learn high-level combinations of features
# Dense (10)	Final output layer: 10 neurons for 10 clothing classes (with softmax activation)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5, callbacks=[MyCallback()])


# Convolutions + pooling layers act like automated feature extractors. 
# They help the network understand the essence of images by compressing 
# and highlighting the most useful patterns — all before reaching the classification layers.