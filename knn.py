from __future__ import division, print_function

from typing import List

import numpy as np
import scipy
from collections import Counter


class KNN:
    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    # save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.X_train = features
        self.y_train = labels

    # predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        X_test = features
        y_test = []
        for point in X_test:
            labels = self.get_k_neighbors(point)
            predicted_label = Counter(labels).most_common(1)[0][0]
            y_test.append(predicted_label)
            
        return y_test

    # find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        distances = []
        labels = []

        for i in range(0, len(self.X_train)):
            distance = self.distance_function(point, self.X_train[i])
            distances.append([distance, i])
        
        distance1 = sorted(distances)

        for i in range(0, self.k):
            index = distance1[i][1]
            labels.append(self.y_train[index])
            
        return labels


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
