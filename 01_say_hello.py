import tensorflow as tf

print('AFTER IMPORT')

h = tf.constant("hello")
sess = tf.Session()
print('AFTER SESSION INIT')

print(sess.run(h))
print('AFTER SESSION')
