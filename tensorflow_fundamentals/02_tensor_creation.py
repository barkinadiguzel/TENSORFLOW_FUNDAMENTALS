import tensorflow as tf

# Constant tensor
a = tf.constant([1, 2, 3])

# Zeros and Ones
zeros = tf.zeros((2,3))
ones = tf.ones((2,2))

# Random tensors
rand_uniform = tf.random.uniform((2,2))  # uniform [0,1)
rand_normal = tf.random.normal((2,2))    # normal mean=0, stddev=1

# Range and linspace
r = tf.range(0,10,2)
ls = tf.linspace(0.0,1.0,5)
