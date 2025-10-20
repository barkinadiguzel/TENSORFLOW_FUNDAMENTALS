import tensorflow as tf

# Set global seed for reproducibility
tf.random.set_seed(42)

# Uniform random numbers between 0 and 1
rand_uniform = tf.random.uniform((2,2))  
# Example output: [[0.3745401, 0.9507143], [0.7319939, 0.5986585]]

# Normal random numbers (mean=0, stddev=1)
rand_normal = tf.random.normal((2,2))
# Example output: [[0.4967141, -0.1382643], [0.64768854, 1.5230298]]

# Note: Running this code multiple times will produce the same numbers due to seed
