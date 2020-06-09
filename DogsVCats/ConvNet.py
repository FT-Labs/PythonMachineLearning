"""
Author : Arda
Date : 5/21/2020
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense , Dropout , Activation, Flatten , Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import os
import numpy as np
import cv2
import time



# NAME = "Cats-vs-dog-cnn-64x2-{}".format(int(time.time()))


y = np.array(pickle.load(open("y.pickle" , "rb")))
X = np.array(pickle.load(open("X1.pickle" , "rb")))
X = X/255.

dense_layers = [0,1,2]
layer_sizes = [32, 64 ,128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer , layer_size , dense_layer , int(time.time()))
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}' , profile_batch=0)
            

            model = Sequential()
            model.add(Conv2D(layer_size , (3,3) , input_shape=(X.shape[1:])))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size= (2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size , (3,3) ))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size= (2,2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))



            # model.add(Dense(64))
            # model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",
                          optimizer="adam",
                          metrics=['accuracy'])

            model.fit(X , y , batch_size=32, validation_split=0.3 , epochs=10 , callbacks=[tensorboard])

# model.save("catdogclassifier.model" , overwrite=True)


cat_0 = cv2.imread("PetImages/Dog/benimkopek2.jpg" , cv2.IMREAD_GRAYSCALE)

cat_0 = cv2.resize(cat_0 , (64,64))
plt.imshow(cat_0 , cmap='gray')
plt.show()
cat_0 = np.array(cat_0).reshape(-1,64,64,1) / 255.


CATEGORIES = ['Cat' , 'Dog']
ans = model.predict(cat_0)

if ans >= 0.5:
    print("Kedi")
else:
    print("Köpek")




