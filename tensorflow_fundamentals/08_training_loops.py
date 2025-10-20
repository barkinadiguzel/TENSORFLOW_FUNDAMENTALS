"""
Simple training loop in TensorFlow with manual gradient update
Includes device placement (GPU if available, else CPU)
"""

import tensorflow as tf

# Sample data
x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y_true = tf.constant([[2.0], [4.0], [6.0], [8.0]])

# Simple linear model
class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable([[0.0]])
        self.b = tf.Variable([0.0])

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

model = LinearModel()

# Loss and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Device setup: GPU if available, else CPU
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    for epoch in range(5):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # if you know pytorch apply gradients is like this:
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        # Simple output to show progress
        # Epoch 0: Loss = 20.0, w = [[0.45]], b = [0.10]
        print(f"Epoch {epoch}: Loss = {loss.numpy():.2f}, w = {model.w.numpy()}, b = {model.b.numpy()}")
