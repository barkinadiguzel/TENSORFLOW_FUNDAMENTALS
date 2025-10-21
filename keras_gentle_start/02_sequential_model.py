"""
Creating a simple Sequential model in Keras.
Demonstrates model creation, compilation, fitting, and evaluation.
"""

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Dummy data
x_train = np.random.rand(100, 20)  # 100 samples, 20 features
y_train = np.random.randint(0, 2, size=(100, 1))  # binary labels

# Sequential model: a simple stack of layers
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(20,)), 
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # output layer for binary classification
])

# Compile the model
model.compile(
    optimizer='adam',          # optimization algorithm
    loss='binary_crossentropy',# loss function for binary classification
    metrics=['accuracy']       # evaluation metric
)

# Fit the model on training data as you see this is more easy with keras
model.fit(x_train, y_train, epochs=5, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_train, y_train)
# loss: e.g., 0.693
# accuracy: e.g., 0.51
