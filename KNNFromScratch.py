from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random

style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

"""
for i in dataset:
    for ii in dataset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)
"""

#[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]       #for visualization of data
#plt.scatter(new_features[0],new_features[1])
#plt.show()

def K_Nearest_Neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total groups!!!')
    """
    Problem with KNN is you need to compare one point to all other ones with the euclidian distance. However,
    it can also be measured with creating circles, then compare the data's in the circle radius. But not for this tutorial.
    That will help you save some time for your computer with calculation.
    """

    distances = list()
    for group in data:
        for feautures in data[group]:
            #euclidian_distance = sqrt((features[0]-predict[0]**2+(feautures[1]-predict[1])**2)) #We wont use this because
            #it takes a lot of time to compute this, also this doesnt work on larger dimensions. Just works on 2D.
            #You need to change the code to adapt it to larger dimensions.
            #euclidian_distance = np.sqrt(np.sum((np.array(feautures)-np.array(predict))**2)) #This will do,
            #however numpy has a simple form for calculating euclidian distances. Down below:
            euclidian_distance = np.linalg.norm(np.array(feautures)-np.array(predict))
            distances.append([euclidian_distance,group])
            votes = [i[1] for i in sorted(distances)[:k]]
            vote_result = (Counter(votes).most_common(1)[0][0])
            Confidence = (Counter(votes).most_common(1)[0][1]) / k

    return vote_result,Confidence

df = pd.read_csv("breast-cancer-wisconsin.txt")
df.replace("?",-99999,inplace=True)
df.drop('id',1,inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data):)]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = K_Nearest_Neighbors(train_set,data,k=5)
        if group==vote:
            correct += 1
        total +=1
Accuracy = correct / total





