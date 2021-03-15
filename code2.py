import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
a=tf.constant([1,2,3])
b=tf.constant([0,0,1])
c=tf.add(a,b)
print(c)

def compute0( a, b, c):
    d = a * b + c
    e = a * b * c
    return d, e

@tf.function 
def compute( a, b, c):
    d = a * b + c
    e = a * b * c
    return d, e

print("++++++++++++")
print(compute0(a,b,c))
print("++++++++++++")
print(compute(a,b,c))
