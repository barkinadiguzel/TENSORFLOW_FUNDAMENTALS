import tensorflow as tf

a = tf.constant([1, 2, 3], dtype=tf.float32)
b = tf.constant([4, 5, 6], dtype=tf.float32)

# Elementwise operations
add = a + b      # [5.0, 7.0, 9.0]
sub = a - b      # [-3.0, -3.0, -3.0]
mul = a * b      # [4.0, 10.0, 18.0]
div = a / b      # [0.25, 0.4, 0.5]
pow_a = tf.pow(a, 2)  # [1.0, 4.0, 9.0]

# Matrix multiplication
mat1 = tf.constant([[1, 2], [3, 4]])
mat2 = tf.constant([[2, 0], [1, 2]])
matmul = tf.matmul(mat1, mat2)
# matmul: [[4, 4], [10, 8]]  # [[1*2+2*1,1*0+2*2],[3*2+4*1,3*0+4*2]]

# Reduction operations
sum_a = tf.reduce_sum(a)      # 6.0
mean_a = tf.reduce_mean(a)    # 2.0
max_a = tf.reduce_max(a)      # 3.0
min_a = tf.reduce_min(a)      # 1.0

# Elementwise functions
sqrt_a = tf.sqrt(a)           # [1.0, 1.414, 1.732]
exp_a = tf.exp(a)             # [2.718, 7.389, 20.085]
log_a = tf.math.log(a)        # [0.0, 0.693, 1.099]  # ln(x)
