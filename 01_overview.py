"""
Keras Overview:
---------------
Keras is a high-level API built on top of TensorFlow.
It makes model building, training, and evaluation much simpler and cleaner.

There are two main ways to build models:
1. Sequential API – For models with a simple layer stack.
2. Functional API – For more flexible or multi-input/multi-output models. for example you can make a input for image and text in same time!!
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sequential API Example

# Used for linear (layer-by-layer) models.

seq_model = keras.Sequential([
    layers.Input(shape=(4,)),           # Input with 4 features
    layers.Dense(8, activation='relu'), # Hidden layer
    layers.Dense(3, activation='softmax')  # Output layer (3 classes)
])

seq_model.summary()
# Sequential model: input → hidden → output
# Used when data flows straight through the network.

# ---------------------------
# Functional API Example
# ---------------------------
# Used when you need multiple inputs, outputs, or layer branching.

inputs = keras.Input(shape=(4,))
x = layers.Dense(8, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(3, activation='softmax')(x)
func_model = keras.Model(inputs=inputs, outputs=outputs)

func_model.summary()
# Functional model allows non-linear architectures and shared layers.
# functional api is like this:
input
  ↓
Dense(64, relu)
  ↓   ↘
skip   Dense(64, relu)
   ↘       ↓
     ← Add() ←
          ↓
Dense(10, softmax)
#but sequentinal ones like
input → Dense → Dense → Dense → output



