import tensorflow as tf


with tf.device('gpu:0'):
    a = tf.constant(1,name='a')
    b = tf.constant(2,name='b')
    c = a * b

print(a.device)
print(b.device)
print(c.device)