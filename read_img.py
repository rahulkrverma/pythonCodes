

import cv2
import numpy as np
import os, sys

def compute_threshold(imgArr, imgHsv, lower, upper):
	state = 0
	mask = cv2.inRange(imgHsv, lower, upper)
	kernel = np.ones((5,5), np.int)
	dilated = cv2.dilate(mask, kernel)
	res = cv2.bitwise_and(imgArr, imgArr, mask=mask)
	ret, thrshed = cv2.threshold(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), 3, 255, cv2.THRESH_BINARY)
	_, contours, hie = cv2.findContours(thrshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	for c in contours:
		if cv2.contourArea(c) < 1000:
			state = 1
	return state

def chheck_solidGreen(imgArr):
	imgHsv = cv2.cvtColor(imgArr, cv2.COLOR_BGR2HSV)
	lower = np.array([80, 80, 80], dtype=np.uint8)
	upper = np.array([189, 255, 255], dtype=np.uint8)
	green = compute_threshold(imgArr, imgHsv, lower, upper)
	return green

def chheck_flashingBlue(imgArr):
	imgHsv = cv2.cvtColor(imgArr, cv2.COLOR_BGR2HSV)
	lower = np.array([65, 105, 255], dtype=np.uint8)
	upper = np.array([138, 43, 226], dtype=np.uint8)
	blue = compute_threshold(imgArr, imgHsv, lower, upper)
	return blue

def chheck_flashingRed(imgArr):
	imgHsv = cv2.cvtColor(imgArr, cv2.COLOR_BGR2HSV)
	lower = np.array([255, 165, 0], dtype=np.uint8)
	upper = np.array([255, 0, 0], dtype=np.uint8)
	red = compute_threshold(imgArr, imgHsv, lower, upper)
	return red

def chheck_flashingYellow(imgArr):
	imgHsv = cv2.cvtColor(imgArr, cv2.COLOR_BGR2HSV)
	lower = np.array([255, 215, 0], dtype=np.uint8)
	upper = np.array([255, 255, 0], dtype=np.uint8)
	yellow = compute_threshold(imgArr, imgHsv, lower, upper)
	return yellow

def chheck_solidBlue(imgArr):
	imgHsv = cv2.cvtColor(imgArr, cv2.COLOR_BGR2HSV)
	lower = np.array([110, 50, 50], dtype=np.uint8)
	upper = np.array([130, 255, 255], dtype=np.uint8)
	sBlue = compute_threshold(imgArr, imgHsv, lower, upper)
	return sBlue

def ring_state(imgArr, imgName):
	green = chheck_solidGreen(imgArr)
	blue = chheck_flashingBlue(imgArr)
	red = chheck_flashingRed(imgArr)
	yellow = chheck_flashingYellow(imgArr)
	sBlue = chheck_solidBlue(imgArr)

	if green == 1:
		print('\n\tImage : \t', imgName, '\t Ring Status : \t Charging Complete')
	elif blue == 1:
		print('\n\tImage : \t', imgName, '\t Ring Status : \t Charging In Progress')
	elif red == 1:
		print('\n\tImage : \t', imgName, '\t Ring Status : \t Charging System Not Activated')
	elif yellow == 1:
		print('\n\tImage : \t', imgName, '\t Ring Status : \t Syatem Check')
	elif sBlue == 1:
		print('\n\tImage : \t', imgName, '\t Ring Status : \t Pause In Charging')
	else:
		print('\n\tImage : \t', imgName, '\t Ring Status : \t Charging System is Fully Switched OFF')


if __name__ == '__main__':
	# 	img_roi = imgArr[200:610 , 350:810]

	imgs = sys.argv[1]
	roi_x = int(sys.argv[2])
	roi_y = int(sys.argv[3])
	roi_h = int(sys.argv[4])
	roi_w = int(sys.argv[5])

	if os.path.isfile(imgs):
		imgArr = cv2.imread(imgs)
		imgRoi = imgArr[roi_x:roi_y, roi_h:roi_w]
		imgName = (imgs.split('/')[-1]).split('.')[0]
		state = ring_state(imgRoi, imgName)
	elif os.path.isdir(imgs):
		imgList = os.listdir(imgs)
		for img in imgList:
			imgArr = cv2.imread(os.path.join(imgs, img))
			imgRoi = imgArr[roi_x:roi_y, roi_h:roi_w]
			imgName = img.split('.')[0]
			state = ring_state(imgRoi, imgName)
	
	


