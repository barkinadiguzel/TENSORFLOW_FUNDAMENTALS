from tensorflow import keras
from tensorflow.keras import layers

# Create a small Sequential model
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(16,)),  
    layers.Dense(10, activation='softmax')                 
])

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save entire model (architecture + weights + optimizer state)
model.save("my_model_full")

# Load full model
loaded_model = keras.models.load_model("my_model_full")

# Save only weights
model.save_weights("my_model_weights.h5")

# Reload weights into the same architecture
new_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(16,)),
    layers.Dense(10, activation='softmax')
])
new_model.load_weights("my_model_weights.h5")
