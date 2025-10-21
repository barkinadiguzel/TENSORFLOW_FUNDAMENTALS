from tensorflow import keras
from tensorflow.keras import layers

# Küçük bir model oluşturuyoruz
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(16,)),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Modeli eğitmeden kaydetme
model.save("my_model_full")  # bu modelin tüm yapısını ve ağırlıklarını kaydeder

# Modeli yükleme
loaded_model = keras.models.load_model("my_model_full")

# Sadece ağırlıkları kaydetmek istersek
model.save_weights("my_model_weights.h5")

# Yüklerken aynı yapıdaki model gerekli
new_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(16,)),
    layers.Dense(10, activation='softmax')
])
new_model.load_weights("my_model_weights.h5")
