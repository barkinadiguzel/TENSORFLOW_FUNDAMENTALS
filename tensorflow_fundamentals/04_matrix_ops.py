import tensorflow as tf

mat = tf.constant([[1, 2, 3], [4, 5, 6]])

# Transpose
mat_T = tf.transpose(mat)
# mat_T: [[1,4],[2,5],[3,6]]

# Reshape
reshaped = tf.reshape(mat, (3,2))
# reshaped: [[1,2],[3,4],[5,6]]

# Broadcasting
vec = tf.constant([10,20,30])
added = mat + vec
# added: [[11,22,33],[14,25,36]]
#--------------------------------------------------------- bottom ones can be little spesific
# Einsum (generalized sum/product)
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[2,0],[1,2]])
einsum_res = tf.einsum('ij,jk->ik', a, b)
# einsum_res: [[4,4],[10,8]]  # same as matmul

# Other linear algebra ops
det = tf.linalg.det(tf.constant([[1.,2.],[3.,4.]]))  # -2.0
inv = tf.linalg.inv(tf.constant([[1.,2.],[3.,4.]]))
# inv: [[-2.,1.],[1.5,-0.5]]
