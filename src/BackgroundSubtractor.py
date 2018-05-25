import cv2
import numpy as np
import copy
import math	
import os
import sys

class BackgroundSubtractor():
	def __init__(self, is_webcam = False):
		# parameters
		self.is_webcam = is_webcam
		self.cap_region_x_begin = 0.5  	# start point/total width
		self.cap_region_y_end = 0.8  	# end point/total width
		self.threshold = 60  		# BINARY threshold
		self.blurValue = 41  		# GaussianBlur parameter
		self.bgSubThreshold = 50

		# variables
		self.isBgCaptured = False   	# bool, whether the background captured
		self.bgModel = None
	
	def clip(self,frame):
		return frame[0:int(self.cap_region_y_end * frame.shape[0]),
			    int(self.cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
		
	def original(self, frame):
		frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
		frame = cv2.flip(frame, 1)  			# flip the frame horizontally
		cv2.rectangle(frame, (int(self.cap_region_x_begin * frame.shape[1]), 0),
			 (frame.shape[1], int(self.cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)	#draw rectangle
		return frame

	def removeBG(self, frame):
		# apply background subtractor and erode
		fgmask = self.bgModel.apply(frame, learningRate = 0.0000)
		kernel = np.ones((3, 3), np.uint8)
		fgmask = cv2.erode(fgmask, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=fgmask)
		return res
         
	def gray_blur_thres(self, frame):
		# grayscale, blur, and threshold
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 0)
		ret, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
		return ret, thresh

	def calculateFingers(self, res, drawing):  # -> finished bool, cnt: finger count
		#  convexity defect
		hull = cv2.convexHull(res, returnPoints=False)
		if len(hull) > 3:
			defects = cv2.convexityDefects(res, hull)
			if type(defects) != type(None):  # avoid crashing.   (BUG not found)
				cnt = 0
				for i in range(defects.shape[0]):  # calculate the angle
					s, e, f, d = defects[i][0]
					start = tuple(res[s][0])
					end = tuple(res[e][0])
					far = tuple(res[f][0])
					a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
					b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
					c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
					angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
					if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
					        cnt += 1
					        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
				return True, cnt
		return False, 0

	def process_frame(self, frame):
		frame = self.original(frame)
		if not self.isBgCaptured 
			if not self.is_webcam:
				cv2.imshow('original', frame)
				k = cv2.waitKey(10)
				if k == 27:  # press ESC to exit
					quit()
				elif k == ord('b'):  # press 'b' to capture the background
					self.bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, self.bgSubThreshold)
					self.isBgCaptured = True
					print('!!!Background Captured!!!')
				return None
			else:
				self.bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, self.bgSubThreshold)
				self.isBgCaptured = True
				return None
			
		else:
			cv2.imshow('original', frame)
			k = cv2.waitKey(10)
			if k == 27:  # press ESC to exit
				quit()
			frame = self.removeBG(frame) 
			frame = self.clip(frame) 
			ret, thresh = self.gray_blur_thres(frame)
			
			return thresh


if __name__ == '__main__':
	cap = cv2.VideoCapture('data/Gen/output_real_1.avi')
	bg = BackgroundSubtractor(True)
	while cap.isOpened():
		ret, frame = cap.read()
		if ret is False:
			break
		frame = bg.process_frame(frame)
		cv2.imshow('binary', frame)
