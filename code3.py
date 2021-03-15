import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
A, B = tf.constant( 3.0), tf.constant( 6.0) 
X = tf.Variable( 20.0) 
loss = tf.math.abs( A * X - B)
def train_step():
    with tf.GradientTape() as tape:
        loss=tf.math.abs(A*X-B)
    dX=tape.gradient(loss,X)
    print(f"X={X.numpy():.2f}, dX={dX:.2f}")
    X.assign(X-dX)
for i in range(7):
    train_step()
