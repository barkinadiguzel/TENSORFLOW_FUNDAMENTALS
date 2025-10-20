# 🔮 TensorFlow Fundamentals Repo

This repository is designed for beginners who want to learn TensorFlow and Keras from scratch.  
You will learn basic concepts, tensor operations, automatic differentiation, model creation, and training loops step by step with examples.

---
## 🧱 Project Structure
```
tensorflow_fundamentals/
├── 01_scalars_vectors_tensors.py        # Tensors, scalars, vectors, shapes, dtype, basic indexing in TensorFlow
├── 02_tensor_creation.py                # Creating tensors: tf.constant, tf.zeros, tf.ones, tf.random, arange, linspace
├── 03_tensor_math_ops.py                # Basic math ops (+, -, *, /), matmul, reduce ops, elementwise functions
├── 04_matrix_ops.py                     # Transpose, reshape, broadcast, einsum, linear algebra operations
├── 05_variables_and_gradients.py        # tf.Variable, gradient recording with tf.GradientTape, basic backprop
├── 06_tf_function_graphs.py             # @tf.function, autograph, converting Python functions into TF graphs
├── 07_modules_layers_models.py          # tf.keras.layers, subclassing tf.keras.Model, building custom layers
├── 08_training_loops.py                 # Simple training loops: forward pass, loss computation, gradient update, optimizer step
├── 09_randomness_and_seeds.py           # Optional: random numbers, reproducibility with tf.random.set_seed
keras_gentle_start/
├── 01_overview.py                       # Keras intro, purpose, Sequential vs Functional overview
├── 02_sequential_model.py               # Small example: Sequential model creation, compile, fit, evaluate
├── 03_functional_api.py                 # Example showing how to connect layers with Functional API
├── 04_saving_loading.py                 # Save and load models using tf.keras.models.save_model / load_model
├── 05_custom_layer_model.py             # Subclassing Layer and Model with a basic example
LICENSE
README.md
```
---
# 📦 Install requirements:
```bash
pip install -r requirements.txt
```
---
## 📬Feedback
For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
