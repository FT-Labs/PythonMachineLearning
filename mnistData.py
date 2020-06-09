"""
Author : Arda
Date : 5/19/2020
"""

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #28x28 images of digits 0-9

(x_train , y_train) , (x_test, y_test) = mnist.load_data()

# plt.imshow(x_train[0])
#To show binary
# plt.imshow(x_train[10] , cmap=plt.cm.binary)
# plt.show()

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)

#Feed-forward model
model = tf.keras.models.Sequential()
#Normalize data
model.add(tf.keras.layers.Flatten())
#Making layers
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128 , activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10 , activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train , y_train , epochs=5)

val_loss , val_acc = model.evaluate(x_test , y_test)

model.save('num_reader.model')

import numpy as np

# pred = model.predict(x_test[6].reshape(x_test[6].shape[0] , -1).T)
pred = model.predict(x_train[647].reshape(1,-1))

print(np.argmax(pred))
print(pred)
plt.imshow(x_train[647])
plt.show()





