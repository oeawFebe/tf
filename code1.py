import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

num_classes = 10 
img_rows, img_cols = 28, 28 
num_channels = 1 
input_shape = (img_rows, img_cols, num_channels) 
(x_train, y_train),( x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0

#Building the model
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,verbose=1,validation_data=(x_test,y_test))

print(model.summary())


