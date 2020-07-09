'''
This Python program takes an image (containing text lines) that is turned. It then calculates the pitch of the text and turns it upright.
This program does not differentiate between text that is turned upside down and text that is upright. It will only turn images in a way
that the new image has text that is completely horizontal.
    
Required packages:
	OpenCV: pip install opencv-python
	Numpy: pip install numpy
	imUtils: pip install imutils
	math: Only import from Python standard library

*** By Luca Lanzo ***
'''

import cv2
import numpy as np
import imutils as mute
import math


def turn_image(image):
	'''
	Checks if the image is on its side and turns it by 90 degrees counterclockwise.
	'''

	height, width, _ = image.shape

	if (height < width):
		return mute.rotate_bound(image, 90)
	else:
		return image


def thresh_image(turnedImage):
	'''
	Converts image from BGR to grayscale
	Then converts grayscale image to binary using an adaptive threshold.
	'''

	grayedImage = cv2.cvtColor(turnedImage, cv2.COLOR_BGR2GRAY)
	threshedImage = cv2.adaptiveThreshold(grayedImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 25)

	return threshedImage


def get_kernel(h, w):
	'''
	This method is critical for the dilation and erosion methods. This kernel represents a box that can be shaped by 
	height and width and will later on determine in which direction and how far the dilation and erosion should go.
	It returns an array with an interchangeable size which serves as a blueprint.
	'''

	return np.ones((h, w), dtype = np.uint8)


def dilate_image(threshedImage):
	'''
	This method dilates the white pixels in the image to make them "bigger" to the left and right. 
	This ensures that the following contour detection won't only detect single letters but rather full lines of text.
	'''

	return cv2.dilate(threshedImage, get_kernel(1, 12))


def erode_image(dilatedImage):
	'''
	This method erodes the lines of text to the top and bottom.
	This ensures that minor pixel deviations to the top or bottom are evened out so multiple lines of text don't 
	connect to each other and form big text blobs.
	'''

	return cv2.erode(dilatedImage, get_kernel(4, 1))


def find_widest_contour(erodedImage):
	'''
	This function will run an algorithm on the binary picture which finds all the contours on each text line.
	To save memory, time and optimize the pitch analysis, only the widest contour will be returned.
	Contours that are too great will be eliminated as only contours that represent text lines are wanted.
	'''
	
	contours, _ = cv2.findContours(erodedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	widestContour = contours[0]

	# find widest contour
	for contour in contours:
		_, _, newContourWidth, newContourHeight = cv2.boundingRect(contour)
		_, _, oldContourWidth, oldContourHeight = cv2.boundingRect(widestContour)

		# Disregard contours which height is far greater than its width
		if newContourHeight > (newContourWidth/100*20):
			continue
		# The widestContour has to have a greater width than the last widest
		elif newContourWidth >= oldContourWidth:
			widestContour = contour
		else:
			continue

	return widestContour


def do_PCA(contour):
	'''
	Calculates the best fitting line for a set of data points using Principal Component Analysis.
	In this context, the set of dataPoints equals the set of pixels of the widest contour.
	The pixels can be interpreted as coordinates in a coordinate system and PCA will find a line that is 
	essentially the median of all points.
	PCA is perfect for reducing dimensionalities. Our sets of pixel are scattered over a coordinate system 
	in a two dimensional manner, while the PCA line is one dimensional.
	'''
	
	dataPoints = np.empty((len(contour), 2), dtype = np.float64)

	for i, dp in enumerate(dataPoints):
		# put the x coordinate of the pixel[i] in the left column of the dataset
		dp[0] = contour[i, 0, 0]
		# put the y coordinate of the pixel[i] in the right column of the dataset
		dp[1] = contour[i, 0, 1]
	
	# Calculate the mean of the contour and then do the PCA
	mean = np.empty((0), dtype = np.float64)
	mean, eigenvectors, eigenvalues = cv2.PCACompute2(dataPoints, mean)

	# Calculate the angle in radians using the arctangent
	angleInRadians = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])

	return angleInRadians


def get_angle_in_degrees(angleInRadians):
	'''
	Converts the angle of the text from radians to degrees.
	'''

	return angleInRadians * 180 / math.pi


def correct_image_alignment(image, angleInDegrees):
	'''
	Corrects the image pitch using the inverse of the calculated angle in degrees 
	'''
	
	return mute.rotate_bound(image, angleInDegrees * (-1))


def align_image(image):
	# turn, thresh, dilate and erode the original image using the preprocessing methods
	preprocessedImage = erode_image(dilate_image(thresh_image(turn_image(image))))

	# find the widest contour
	widestContour = find_widest_contour(preprocessedImage)

	# find a best fit line for the widest contour using PCA
	angleInRadians = do_PCA(widestContour)

	# convert the angle of the best fit line from radians to degrees
	angleInDegrees = get_angle_in_degrees(angleInRadians)
	
	# reverse the image rotation using the calculated pitch
	alignedImage = correct_image_alignment(image, angleInDegrees)

	return alignedImage
