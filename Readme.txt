

HUMAN DETECTION USING HOG - README


Files

Pre-trained models are present in "model" directory.

Some sample output images are present in "OP_images" directory.

Report-HumanDetection.pdf is the project report file.

custom_hog_function.py: This file contains the implementation of our custom HOG (Histogram of Oriented Gradients) feature extractor.

For training/testing images
https://drive.google.com/drive/folders/1nvBuVXTuNzGPwbvSFmg-U4PVu6-Qc_QU?usp=sharing


$Training$
1. custom_Train_HOG_SVM.py: Use this script to train an SVM (Support Vector Machine) classifier based on the extracted HOG features.

2. Perceptron_implementation.py: This script is designed for training a Perceptron classifier using the extracted HOG features.

Note: Training should be performed using images from the "1-train" and "0-train" folders.

$Testing$
1. custom_testing_HOG_SVM.py: Employ this script to test a given image using the HOG+SVM classifier. Ensure to provide the test image path at line 36 in the script.

2. custom_testing_perceptron.py: Use this script to test a given image using the HOG+Perceptron classifier. Provide the test image path at line 38 in the script.


Note: Adjust the paths accordingly for seamless testing.

Happy coding! 
