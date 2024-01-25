import joblib
import numpy as np
import os
import glob
from skimage.feature import hog
from skimage import color
import cv2
from sklearn.preprocessing import LabelEncoder



# class Perceptron:
#     def __init__(self, learning_rate=0.01, epochs=1000, num_features=3780):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.weights = None
#         self.bias = 0.0 

#     def train(self, X, y):
#         num_samples, num_features = X.shape if len(X.shape) > 1 else (X.shape[0], 1)
#         self.weights = np.zeros((num_features, 1))  # Updated line to create a column vector
#         for epoch in range(self.epochs):
#             for i in range(num_samples):
#                 y_pred = np.sign(np.dot(X[i], self.weights) + self.bias)

#                 if y_pred != y[i]:
#                     self.weights += self.learning_rate * y[i] * X[i]
#                     self.bias += self.learning_rate * y[i]


#     def predict(self, X):
#         # print("predict.np",np.atleast_2d(X))
#         X = np.atleast_2d(X)

#         # Perform prediction
        
#         return X


#     def save_model(self, filename="perceptron_model.npy"):
#         joblib.dump({"weights": self.weights, "bias": self.bias}, filename)

#     def load_model(self, filename="perceptron_model.npy"):
#         loaded_data = joblib.load(filename)
#         self.weights = loaded_data["weights"]
#         self.bias = loaded_data["bias"]


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000, num_features=3780):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def train(self, X, y):
        num_samples, num_features = X.shape if len(X.shape) > 1 else (X.shape[0], 1)
        self.weights = np.zeros((num_features, 1))
        for epoch in range(self.epochs):
            for i in range(num_samples):
                y_pred = np.sign(np.dot(X[i], self.weights) + self.bias)

                if y_pred != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        # Ensure X is a 2D array
        X = np.atleast_2d(X)

        # Perform prediction

        return X
      

    def save_model(self, filename="perceptron_model.npy"):
        joblib.dump({"weights": self.weights, "bias": self.bias}, filename)

    def load_model(self, filename="perceptron_model.npy"):
        loaded_data = joblib.load(filename)
        self.weights = loaded_data["weights"]
        self.bias = loaded_data["bias"]
