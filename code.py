import cv2
import numpy as np
import os

#lists all the image files in the folder
relevant_path = "E:\img_reg"
included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
file_names = [fn for fn in os.listdir(relevant_path)
              if any(fn.endswith(ext) for ext in included_extensions)]


#naming the output files
output_files=[]
for i in range(0,len(file_names)):
	output_files.append("output"+str(i)+".jpg")

for j in range(0,len(file_names)):

	# Open the image files.
	img1_color = cv2.imread(file_names[j]) # Image to be aligned.
	img2_color = cv2.imread("original.jpg") # Reference image.

	# Convert to grayscale.
	img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
	height, width = img2.shape

	# Create ORB detector with 5000 features.
	orb_detector = cv2.ORB_create(5000)

	# Find keypoints and descriptors.
	# The first arg is the image, second arg is the mask
	# (which is not reqiured in this case).
	kp1, d1 = orb_detector.detectAndCompute(img1, None)
	kp2, d2 = orb_detector.detectAndCompute(img2, None)

	# Match features between the two images.
	# We create a Brute Force matcher with
	# Hamming distance as measurement mode.
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

	# Match the two sets of descriptors.
	matches = matcher.match(d1, d2)

	# Sort matches on the basis of their Hamming distance.
	matches.sort(key = lambda x: x.distance)

	# Take the top 90 % matches forward.
	matches = matches[:int(len(matches)*90)]
	no_of_matches = len(matches)

	# Define empty matrices of shape no_of_matches * 2.
	p1 = np.zeros((no_of_matches, 2))
	p2 = np.zeros((no_of_matches, 2))

	for i in range(len(matches)):
		p1[i, :] = kp1[matches[i].queryIdx].pt
		p2[i, :] = kp2[matches[i].trainIdx].pt

	# Find the homography matrix.
	homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

	# Use this matrix to transform the
	# colored image wrt the reference image.
	transformed_img = cv2.warpPerspective(img1_color,
						homography, (width, height))

	# Save the output.
	folder="Output"+chr(92)
	cv2.imwrite(folder+output_files[j], transformed_img)
