import tensorflow as tf

# input tensor
a = tf.constant(1.0)
b = tf.Variable(2.0)
c = tf.placeholder(dtype=tf.float32, name='x')

# make session
sess = tf.Session()
print("\n\n\n\n\n")

# operation (operation overloading can be applied)
add_all = a + b + c

# (if variable is used, plese activate initializor first)
init = tf.global_variables_initializer()
sess.run(init)

# session run
r = sess.run(add_all, feed_dict={c: [3, 1]})
print(r)

# window : ctrl + `
# mac : ctrl + shift + `
