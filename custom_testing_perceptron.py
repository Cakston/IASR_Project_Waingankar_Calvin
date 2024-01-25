from skimage.feature import hog
from skimage.transform import pyramid_gaussian
#from sklearn.externals 
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
from sklearn.metrics import classification_report
import imutils
import numpy as np
import cv2
import os
import glob
import custom_hog_function

#Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])
#%%
# Upload the saved svm model:
# model = joblib.load("perceptron_model.npy")
model = joblib.load('perceptron_model.npy')

# Test the trained classifier on an image below!
scale = 0
detections = []
# read the image you want to detect the object in:
img= cv2.imread("archive/human detection dataset/1 - Copy/235.png") #,cv2.IMREAD_GRAYSCALE)
#2,3,6,34,342
#422
#123,200,240
# Try it with image resized if the image is too big
img= cv2.resize(img,(300,200)) # can change the size to default by commenting this code out our put in a random number

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.5
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=1.5):
    for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Ensure the window has the correct number of channels (3 for RGB)
        if window.shape[-1] != 3:
            print("Warning: Window does not have 3 channels. Skipping window.")
            continue

        fds = custom_hog_function.hog(window)

        if np.isnan(fds).any():
            print("Warning: NaN values found in HOG features. Skipping window.")
            continue

        if np.all(fds == 0):
            print("Warning: All-zero HOG features. Skipping window.")
            continue

        fds = fds.reshape(1, -1)

        fds_mask = ~np.isnan(fds)
        fds = fds[fds_mask]

        if fds.shape[0] == 0:
            print("Warning: Empty feature vector. Skipping window.")
            continue
        # print("fds",fds.size)


        pred = model.predict(fds)
        decision_values = model.decision_function(fds) if hasattr(model, 'decision_function') else pred 
 # Use decision_function if available, else use pred 

        # print("Shape of fds before prediction:", fds.shape)
        # print("pred", pred)
        print("Decision Values:", decision_values)

            # Store all detections with their decision values
        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), decision_values[0],
                           int(windowSize[0]*(downscale**scale)), int(windowSize[1]*(downscale**scale))))

    scale += 1
    
    
clone = resized.copy()
# for (x_tl, y_tl, _, w, h) in detections:
#     cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
sc = [score[y] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.2)
print("detection confidence pick: ", pick)
# the peice of code above creates a raw bounding box prior to using NMS
# the code below creates a bounding box after using nms on the detections
# you can choose which one you want to visualise, as you deem fit... simply use the following function:
# cv2.imshow in this right place (since python is procedural it will go through the code line by line).
        
for (xA, yA, xB, yB) in pick:
 
    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # for (x, y, confidence, w, h) in detections:
    #     print("Detection at ({}, {}): Confidence Score: {}".format(x, y, confidence))

    cv2.imshow("Raw Detections after NMS", img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite("of_saved_image.png", img)
        cv2.destroyAllWindows()
        
    cv2.imshow("Raw Detections after NMS", img)
    #### Save the images below
    k = cv2.waitKey(0) & 0xFF 
    if k == 27:             #wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite("of_saved_image.png",img)
        cv2.destroyAllWindows()