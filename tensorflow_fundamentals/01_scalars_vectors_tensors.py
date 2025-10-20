import tensorflow as tf

# Scalar: 0D tensor (single number)
scalar = tf.constant(5)  # dtype inferred as int32

# Vector: 1D tensor
vector = tf.constant([1, 2, 3])

# Matrix: 2D tensor
matrix = tf.constant([[1, 2], [3, 4]])

# Higher-dimensional tensor: 3D example
tensor = tf.constant([[[1],[2]], [[3],[4]]])

# Indexing examples
# Access first row, second element of matrix
element = matrix[0,1]

# Slice vector starting from second element
sub_vector = vector[1:]
