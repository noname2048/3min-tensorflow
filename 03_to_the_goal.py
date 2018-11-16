import tensorflow as tf


# data setting
x_d = [1, 2, 3]
y_d = [1, 2, 3]

# place holder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Variable
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost) # how about maximize?


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        #but optimizer return nothing
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_d, Y: y_d})
        print(step, cost_val, sess.run(W), sess.run(b))
        
    print(sess.run(hypothesis, feed_dict={X: 5}))
