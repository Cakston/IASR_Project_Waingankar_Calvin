import os
import glob
import numpy as np
from skimage.feature import hog
from skimage import color
import cv2
from perceptron import Perceptron
import joblib
from sklearn.preprocessing import LabelEncoder

# Assuming you have separate directories for positive and negative samples
positive_samples_path = "archive/human detection dataset/1 - train"
negative_samples_path = "archive/human detection dataset/0 - train"


# Create a list to store perceptron models
perceptron_models = []

# Load positive samples
positive_samples = []
for file_path in glob.glob(os.path.join(positive_samples_path, "*.png")):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (300, 200))  # Adjust size as needed
    positive_samples.append(img)

# Load negative samples
negative_samples = []
for file_path in glob.glob(os.path.join(negative_samples_path, "*.png")):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (300, 200))  # Adjust size as needed
    negative_samples.append(img)

# Extract HOG features and create labeled training data
X_train = []
y_train = []

# Positive samples
for img in positive_samples:
    img_gray = color.rgb2gray(img)
    fds = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
    X_train.append(fds)
    y_train.append(1)  # Positive label

# Negative samples
for img in negative_samples:
    img_gray = color.rgb2gray(img)
    fds = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2', feature_vector=True)
    X_train.append(fds)
    y_train.append(0)  # Negative label


le = LabelEncoder()
labels = le.fit_transform(y_train)


# Convert lists to numpy arrays
X_train = np.array(X_train)
# y_train = np.array(y_train)

# Train multiple perceptrons
    # Create an instance of Perceptron
perceptron_model = Perceptron()

    # Train the perceptron
perceptron_model.train(X_train, y_train)

    # Save the trained model
# perceptron_model.save_model(f"perceptron_model.npy")
joblib.dump(perceptron_model, "perceptron_model.npy")

    # Append the trained model to the list
perceptron_models.append(perceptron_model)

print("perceptron_model",perceptron_model)