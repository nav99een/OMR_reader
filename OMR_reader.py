#run python3 OMR_reader.py --image images/example_test.png

#importing libraries that will be used in this script
import numpy as np
import cv2
import argparse
import imutils 
from imutils import contours
from imutils.perspective import four_point_transform

#Reading argumnets from command line or bash
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True, help='path to the image')
args = vars(ap.parse_args())

#defining actual answers
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

#reading image
img = cv2.imread(args['image'])
orig = img.copy()

#converting to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Taking gaussian blur to remove noise from image
blur = cv2.GaussianBlur(gray,(5,5),0)
edge = cv2.Canny(blur,75,200)

cv2.imshow('edge',edge)
cv2.waitKey(0)
cv2.destroyAllWindows()

#finding contours from edge image
cnts = cv2.findContours(edge.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
doCnts = None

if len(cnts):
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx)==4:
			doCnts=approx
			break

#correcting the orientation of the image if not scanned correctly
paper = four_point_transform(img,doCnts.reshape(4,2))
warp = four_point_transform(gray,doCnts.reshape(4,2))

#Thresholding the image
thresh = cv2.threshold(warp, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
	(x,y,w,h) = cv2.boundingRect(c)
	ar = w/float(h)
	
	if w>=20 and h>=20 and ar>=0.9 and ar<=1.1:
		questionCnts.append(c)

questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]

correct = 0

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	# sort the contours for the current question from
	# left to right, then initialize the index of the
	# bubbled answer
	cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
	bubbled = None

	# loop over the sorted contours
	for (j, c) in enumerate(cnts):
		# construct a mask that reveals only the current
		# "bubble" for the question
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		# apply the mask to the thresholded image, then
		# count the number of non-zero pixels in the
		# bubble area
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)

		# if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# initialize the contour color and the index of the
	# *correct* answer
	color = (0, 0, 255)
	k = ANSWER_KEY[q]

	# checking to see if the bubbled answer is correct
	if k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# drawing the outline of the correct answer on the test
	cv2.drawContours(paper, [cnts[k]], -1, color, 3)



score = (correct/5.0)*100

#putting score
cv2.putText(paper,"{:.2f}%".format(score),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

cv2.imshow('original',orig)

cv2.imshow('scored',paper)

cv2.waitKey(0)





#================*****============*****============****=============END==========*****============****==========***===========*****==============#
