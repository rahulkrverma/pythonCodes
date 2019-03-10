


import cv2
import numpy as np
import imutils

def get_img(img_arr):

	img_arr = imutils.resize(img_arr, height = 500)

	img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.bilateralFilter(img_gray, 11, 17, 17)
	img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
	img_edge = cv2.Canny(img_gray, 30, 200)

	_, cnts, _ = cv2.findContours(img_edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	outline = np.zeros(img_arr.shape, dtype = "uint8")
	
	cv2.drawContours(outline, cnts, -1, (255, 255, 255), -1)

	return outline


if __name__ == '__main__':

	img_1 = cv2.imread('thresh2.jpg')
	img_2 = cv2.imread('thresh3.jpg')

	con_img1 = get_img(img_1)
	con_img2 = get_img(img_2)

	cv2.imwrite('processed_th2.jpg', con_img1)
	cv2.imwrite('processed_th3.jpg', con_img2)
