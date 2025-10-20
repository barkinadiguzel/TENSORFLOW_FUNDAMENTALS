import tensorflow as tf

# Creating a variable
x = tf.Variable([1.0, 2.0, 3.0])  # mutable tensor
# x can be updated
x.assign([4.0, 5.0, 6.0])  
# x: [4.0, 5.0, 6.0]

# Automatic differentiation
y = tf.Variable([2.0, 3.0, 4.0])

with tf.GradientTape() as tape:
    z = y * y * 2  # z = 2 * y^2

dz_dy = tape.gradient(z, y)  
# dz_dy: [8.0, 18.0, 32.0]  # derivative of 2*y^2 w.r.t y

# Updating variables using gradients manually
learning_rate = 0.1
y.assign_sub(learning_rate * dz_dy)  
# y: [1.2, 1.2, 0.8]  # you can imagine y act like weight or parameter
