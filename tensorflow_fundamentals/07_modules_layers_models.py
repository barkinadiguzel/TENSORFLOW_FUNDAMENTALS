import tensorflow as tf

# Step 1: Define a custom model by subclassing tf.keras.Model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers
        self.dense1 = tf.keras.layers.Dense(4, activation='relu')  # hidden layer
        self.dense2 = tf.keras.layers.Dense(2)                     # output layer

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        return self.dense2(x)

# Step 2: Instantiate the model
model = MyModel()

# Step 3: Create some dummy input data
x = tf.constant([[1.0, 2.0, 3.0]])

# Step 4: Run forward pass
y = model(x)

# y is the output tensor
# Example output (values will vary due to random initialization):
# y: [[0.12, -0.34]]
