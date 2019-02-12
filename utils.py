import numpy as np
from typing import List
from knn import KNN

def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:   
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for i in range(len(real_labels)):
        if predicted_labels[i] == 1:
            if real_labels[i] == predicted_labels[i]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if real_labels[i] == predicted_labels[i]:
                true_negatives += 1
            else:
                false_negatives += 1
    
    if true_positives == 0:
        return 0
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 / ((1 / precision) + (1 / recall))
    
    return f1_score


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    return np.linalg.norm(np.array(point1) - np.array(point2))


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    return np.inner(np.array(point1), np.array(point2))


def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    return -np.exp(-0.5*((np.linalg.norm(np.array(point1) - np.array(point2)))**2))


def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    point1_array = np.array(point1)
    point2_array = np.array(point2)   
    return 1 - np.dot(point1_array, point2_array) / (np.sqrt(np.dot(point1_array, point1_array)) * np.sqrt(np.dot(point2_array, point2_array)))


# selects an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    
    best_k = 0
    best_model = 0
    best_function = 0
    best_f1_score = 0
    limit = 30
    if limit > len(ytrain):
        limit = len(ytrain)
        
    for function_key, function_value in distance_funcs.items():
        for k in range(1, limit):      
            knn = KNN(k, function_value)
            knn.train(Xtrain, ytrain)
            predicted_labels = knn.predict(Xval)
            real_labels= yval
            f1 = f1_score(real_labels, predicted_labels)

            if f1 > best_f1_score:
                best_k = k
                best_model = knn
                best_function = function_key
                best_f1_score = f1
    
#     print(best_f1_score)
    return best_model, best_k, best_function


# selects an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    
    best_k = 0
    best_model = 0
    best_function = 0
    best_f1_score = 0
    best_scaler = 0
    
    for scaler_key, scaler_value in scaling_classes.items():
        scaler = scaler_value()
        Xtrain = scaler(Xtrain)
        Xval = scaler(Xval)
        limit = 30
        if limit > len(ytrain):
            limit = len(ytrain) - 1

        for function_key, function_value in distance_funcs.items():               
            for k in range(1, limit):      
                knn = KNN(k, function_value)
                knn.train(Xtrain, ytrain)
                predicted_labels = knn.predict(Xval)
                real_labels= yval
                f1 = f1_score(real_labels, predicted_labels)

                if f1 > best_f1_score:
                    best_k = k
                    best_model = knn
                    best_function = function_key
                    best_f1_score = f1
                    best_scaler = scaler_key

    return best_model, best_k, best_function, best_scaler


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        normalised_data = []

        for feature in features:
            numerator = np.array(feature)
            denominator = np.sqrt(np.inner(np.array(feature), np.array(feature)))
            if denominator == 0:
                normalised_data.append(feature)
            else:
                normalised_data.append((numerator/denominator).tolist())
        return normalised_data


class MinMaxScaler:
    def __init__(self):
        self.min_max_data = []
        self.training_data_scaled = False

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        np_array = np.array(features)
               
        if not self.training_data_scaled:
            for i in range(0, np_array.shape[1]):
                c_max = np_array.max(axis=0)[i]
                c_min = np_array.min(axis=0)[i]
                self.min_max_data.append([c_max, c_min])             
            self.training_data_scaled = True            
              
        arr_1 = []
        for i in range(0, np_array.shape[0]):
            arr_2 = []
            for j in range(0, np_array.shape[1]):
                xi = np_array[i][j]
                xmax = self.min_max_data[j][0]
                xmin = self.min_max_data[j][1]
                numerator = xi - xmin
                denominator = xmax - xmin
                if numerator == 0 or denominator == 0:
                    arr_2.append(0)
                else:
                    arr_2.append(numerator/denominator)
            arr_1.append(arr_2)

        return arr_1
