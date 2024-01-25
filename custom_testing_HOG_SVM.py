from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
import custom_hog_function

# Define HOG Parameters
# Change them if necessary: orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = 0.3

# Define the sliding window function
def sliding_window(image, stepSize, windowSize):
    # Slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # Yield the current window
            yield (x, y, image[y: y + windowSize[1], x: x + windowSize[0]])

# Upload the saved SVM model
model = joblib.load("model_SVM.npy")

# Test the trained classifier on an image
scale = 0
detections = []

# Read the image for object detection
img = cv2.imread("C:/Users/waing/OneDrive - Politechnika Warszawska/Dokumenty/IASRProject/iasr_project/archive/human detection dataset/test-table/545.png") #76,80,100,134,174,208,212,213,247,250,294,336,347,353,363,416,429,509,520,542.

# Resize the image (optional)
img = cv2.resize(img, (300, 200))

# Define the size of the sliding window (same as the size in the training data)
(winW, winH) = (64, 128)
windowSize = (winW, winH)
downscale = 1.5

# Apply sliding window using Gaussian pyramid
for resized in pyramid_gaussian(img, downscale=1.5):
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW or window.shape[2] != 3:
            continue

        # Extract HOG features from the window
        fds = custom_hog_function.hog(window)
        fds = fds.reshape(1, -1)

        # Remove NaN values from HOG features
        fds_mask = ~np.isnan(fds).any(axis=1)
        fds = fds[fds_mask]

        # Make a prediction using the SVM model
        pred = model.predict(fds)

        if pred == 1 and model.decision_function(fds) > 0.75:
            # Record the detection coordinates and confidence score
            detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                               int(windowSize[0] * (downscale**scale)), int(windowSize[1] * (downscale**scale))))
    scale += 1

# Create a clone of the resized image for visualization
clone = resized.copy()

# Draw rectangles around the detected objects
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
# print(sc,"sc")
# Perform non-maximum suppression (NMS) to remove overlapping rectangles
sc = np.array(sc)
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.1)

# Draw the final rectangles after NMS
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    cv2.imshow("Raw Detections after NMS", img)

    # Save the image on key press 's'
    k = cv2.waitKey(0) & 0xFF
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite("of_saved_image.png", img)
        cv2.destroyAllWindows()
