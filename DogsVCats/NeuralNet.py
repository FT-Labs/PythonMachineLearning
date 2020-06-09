"""
Author : Arda
Date : 5/25/2020
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

y = np.array(pickle.load(open("y.pickle" , "rb")))
X = np.array(pickle.load(open("X.pickle" , "rb")))
X = X/255.

model = Sequential()

model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X , y , batch_size=32 , validation_split=0.2 , epochs=10)



