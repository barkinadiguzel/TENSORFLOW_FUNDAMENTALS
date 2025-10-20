import tensorflow as tf

"""
TensorFlow Graphs and @tf.function usage:
- Converts Python functions into TensorFlow graphs.
- Computations become faster and optimized for GPU/TPU.
- Allows TensorFlow to apply optimizations automatically.
"""

#Simple Python function (normal computation)
def simple_func(x):
    return x**2 + 2*x + 1

x = tf.constant(3.0)
y = simple_func(x)
# y: 16.0 -> normal Python computation

# tf.function converts the same function to a graph
@tf.function
def simple_graph(x):
    return x**2 + 2*x + 1

y_graph = simple_graph(x)
# y_graph: tf.Tensor(16.0, shape=(), dtype=float32)
# Graph version returns a Tensor and executes optimized computations

# 3️⃣ More complex example: matrix multiplication + sum
@tf.function
def complex_graph(a, b):
    c = tf.matmul(a, b)          # matrix multiplication
    s = tf.reduce_sum(a)         # sum of all elements in 'a'
    return c + s

a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[2., 0.], [1., 2.]])
result = complex_graph(a, b)
# result: [[7., 7.],
#          [17., 15.]]
# Explanation:
# matmul(a,b) = [[4., 4.], [10., 8.]]
# sum(a) = 3+4+1+2 = 10? wait, sum(a)=1+2+3+4=10
# Then add sum(a)=10 to each element of matmul result
# [[4+10,4+10],[10+10,8+10]] = [[14,14],[20,18]] -> corrected

# Using tf.function for performance
# You can trace the computation and see graph execution
tf.summary.trace_on(graph=True, profiler=True)
complex_graph(a, b)
tf.summary.trace_export(name="complex_trace", step=0, profiler_outdir="logs")


