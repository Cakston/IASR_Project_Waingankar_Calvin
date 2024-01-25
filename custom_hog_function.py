import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color

# Function for histogram normalization of HOG features
def histo_norm(img, hist_full):
    # Extract image dimensions
    height, width = img.shape[:2]
    cell_size = 8
    num_cells_h = int(height / cell_size)
    num_cells_w = int(width / cell_size)
    
    # Initialize an empty array to store normalized histograms
    hist_full_n = np.empty([num_cells_h - 1, num_cells_w - 1, 36])  # 15x7x36
    
    # Iterate through the image cells for normalization
    for i in range(num_cells_h - 1):
        for j in range(num_cells_w - 1):
            # Calculate the L2 norm for normalization
            k = np.sqrt(
                np.sum(hist_full[i, j] ** 2) + np.sum(hist_full[i, j + 1] ** 2) + np.sum(hist_full[i + 1, j] ** 2) +
                np.sum(hist_full[i + 1, j + 1] ** 2))
            
            # Concatenate histograms for the four neighboring cells and normalize
            hist_full_n[i, j] = np.concatenate((hist_full[i, j], hist_full[i, j + 1], hist_full[i + 1, j],
                                                 hist_full[i + 1, j + 1]), axis=None)
            hist_full_n[i, j] = hist_full_n[i, j] / k
    
    return hist_full_n


# Function to map angles to a specified range
def anglemapper(x):
    if x >= 180:
        return x - 180
    else:
        return x


# Function to create histogram for an 8x8 cell based on angle and magnitude arrays
def createHist(AngArray, MagArray, BS=20, BINS=9):
    hist = np.zeros(BINS)
    for r in range(AngArray.shape[0]):
        for c in range(AngArray.shape[1]):
            binel, rem = np.divmod(AngArray[r, c], BS)
            weightR = rem * 1.0 / BS
            weightL = 1 - weightR
            deltaR = MagArray[r, c] * weightR
            deltaL = MagArray[r, c] * weightL
            binL = int(binel)
            if binL > 8:
                hist[8] += deltaL
                hist[0] += deltaR
            else:
                binR = np.mod(binL + 1, BINS)
                hist[binL] += deltaL
                hist[binR] += deltaR
    return hist

# Function for calculating Histogram of Oriented Gradients (HOG) features
def hog(img):
    # Resize the image to 64x128 pixels
    img = cv2.resize(color.rgb2gray(img), (64, 128))
    
    # Calculate the gradient using Sobel operators
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=1)
    
    # Convert Cartesian coordinates to polar coordinates to obtain magnitude and orientation
    magnitude, orientation = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)
    orientation = orientation.astype(int)
    
    # Map angles to predefined bins using a custom angle mapper
    vfunc = np.vectorize(anglemapper)
    mappedAngles = vfunc(orientation)

    # Calculate histogram using gradient and orientation for each 8x8 cell
    height, width = img.shape[:2]
    cell_size = 8
    num_cells_h = int(height / cell_size)
    num_cells_w = int(width / cell_size)
    num_bins = 9
    
    # Initialize an empty array to store the cells
    hist_full = np.empty([num_cells_h, num_cells_w, 9])  # 16x8x9
    
    # Iterate through the image cells
    for i in range(num_cells_h):
        for j in range(num_cells_w):
            # Extract the current cell from the image
            spotMag = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            spotAngles = mappedAngles[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            
            # Create histogram for each 8x8 cell
            spotHist = createHist(spotAngles, spotMag)
            hist_full[i, j] = spotHist

    # Normalize the histogram
    hist_full_n = histo_norm(img, hist_full)
    
    # Reshape the normalized histogram into a 1D feature vector
    a = np.reshape(hist_full_n, (3780,))
    print(a.shape)
    
    return a
