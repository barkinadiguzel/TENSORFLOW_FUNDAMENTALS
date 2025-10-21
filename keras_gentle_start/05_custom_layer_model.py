from tensorflow import keras
from tensorflow.keras import layers

# Custom layer
class MyDenseLayer(layers.Layer):
    def __init__(self, units=32):
        super(MyDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        #define weight and bias
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return keras.activations.relu(tf.matmul(inputs, self.w) + self.b)
# ACTUALLY ITS SAME WİTH layer = "layers.Dense(32, activation='relu')" but if you wanna change in inside you need to know the sample too.
# Custom model
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = MyDenseLayer(64)
        self.d2 = MyDenseLayer(10)

    def call(self, inputs):
        x = self.d1(inputs)
        return self.d2(x)

# Create the model
model = MyModel()

# Compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
