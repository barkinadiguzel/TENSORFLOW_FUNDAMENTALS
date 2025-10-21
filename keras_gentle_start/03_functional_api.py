"""
Demonstrating Keras Functional API to build complex models
with multiple inputs/outputs or non-linear architectures.
"""

from tensorflow import keras
from tensorflow.keras import layers

# Input layer
inputs = keras.Input(shape=(32,))  # input vector of size 32

# First dense layer with ReLU
x = layers.Dense(64, activation='relu')(inputs)

# Second dense layer
y = layers.Dense(64, activation='relu')(x)

# Example of residual connection (skip connection)
residual = layers.Add()([x, y])  # adds output of first layer to second, you can't do this with seq. model

# Output layer
outputs = layers.Dense(10, activation='softmax')(residual)

# Define the model
func_model = keras.Model(inputs=inputs, outputs=outputs)

# Compile model
func_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Model summary
func_model.summary()
