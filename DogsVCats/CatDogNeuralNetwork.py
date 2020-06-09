"""
Author : Arda
Date : 5/20/2020
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "PetImages"
CATEGORIES = ['Dog' , 'Cat']



training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # cats or dogs path
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # plt.imshow(img_array, cmap='gray')
                # plt.show()
                IMG_SIZE = 64
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resizing shapes to 50x50
                training_data.append([new_array , class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = []
y = []

for features , label in training_data:
    X.append(features)
    y.append(label)

#For reshaping arrays to 1dim
X =  np.array(X).reshape(-1 ,64 , 64 , 1)

pickle_out = open("X1.pickle" , "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle" , "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# print(X[1])
# print(X.shape[0])
# print(X.shape[1])