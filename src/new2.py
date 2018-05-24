import cv2
import numpy as np
import copy
import math	
import os
import threading
import sys

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))

class FingerCounter():
	def __init__(self, video_dir = None):
		self.save = False
		self.video_dir = video_dir
		self.use_video = video_dir is not None

		# parameters
		self.cap_region_x_begin = 0.5  # start point/total width
		self.cap_region_y_end = 0.8  # start point/total width
		if self.use_video:
			self.cap_region_x_begin = 0  # start point/total width
			self.cap_region_y_end = 1  # start point/total width
		self.threshold = 60  #  BINARY threshold
		self.blurValue = 41  # GaussianBlur parameter
		self.bgSubThreshold = 50

		# variables
		self.isBgCaptured = 0   # bool, whether the background captured
		self.triggerSwitch = False  # if true, keyborad simulator works

		self.camera = None
		self.fourcc = cv2.VideoWriter_fourcc(*'XVID') 
		self.video_num = None
		self.prev_cnt = -1

		self.writer_bin = None
		self.writer_real = None

		self.bgModel = None
	
	def removeBG(self, frame):
		fgmask = self.bgModel.apply(frame, learningRate = 0.0000)
		kernel = np.ones((3, 3), np.uint8)
		fgmask = cv2.erode(fgmask, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=fgmask)
		return res
         
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

	def init_camera(self):
		# Camera
		if self.video_dir is None:
			self.camera = cv2.VideoCapture(0)
		else:
			self.camera = cv2.VideoCapture(self.video_dir)
			self.use_video = True
		self.camera.set(10,100)
		cv2.namedWindow('trackbar')
		cv2.createTrackbar('trh1', 'trackbar', self.threshold, 100, printThreshold)
		
	def init_save_video(self):
		if self.save:
			with open('../data/video/label.txt','r') as f:
				self.video_num = int(f.readline())
			self.writer_bin = cv2.VideoWriter('../data/video/output_bin_' + str(self.video_num) + '.avi',self.fourcc, 20.0, (320,384))
			self.writer_real = cv2.VideoWriter('../data/video/output_real_'+str(self.video_num)+ '.avi', self.fourcc, 20.0, (320,384))

	def quit_save_video(self):
		if self.save:
			self.writer_bin.release()
			self.writer_real.release()
			with open('../data/video/label.txt','w') as f:
				f.write(str(self.video_num))
		

	def init_game(self):
		if self.use_video:
			ret, frame = self.camera.read()
			self.bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, self.bgSubThreshold)
			self.isBgCaptured = 1
			
		while self.camera.isOpened() and not self.use_video:
			ret, frame = self.camera.read()
			self.threshold = cv2.getTrackbarPos('trh1', 'trackbar')
			frame = self.original(frame)
			cv2.imshow('original',frame)
			k = cv2.waitKey(10)
			if k == 27:  # press ESC to exit
				sys.exit()
				self.camera.release()
			elif k == ord('b'):  # press 'b' to capture the background
				self.init_save_video()
				self.bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, self.bgSubThreshold)
				self.isBgCaptured = 1
				print('!!!Background Captured!!!')
				break

	def original(self, frame):
		frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
		frame = cv2.flip(frame, 1)  # flip the frame horizontally
		cv2.rectangle(frame, (int(self.cap_region_x_begin * frame.shape[1]), 0),
			 (frame.shape[1], int(self.cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
		#cv2.imshow('original', frame)

		return frame

	def clip(self,img):
		return img[0:int(self.cap_region_y_end * img.shape[0]),
			    int(self.cap_region_x_begin * img.shape[1]):img.shape[1]]  # clip the ROI
		
	def gray_blur_thres(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 0)
		#cv2.imshow('blur', blur)
		ret, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
		return ret, thresh
		
	def write_video(self, frame, thresh):
		if self.save:
			self.writer_real.write(frame)
			towrite = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
			self.writer_bin.write(towrite)

	def get_contour(self, thresh):
		thresh1 = copy.deepcopy(thresh)
		_, contours, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		length = len(contours)
		return contours, length
		

	def play_game(self):
		while self.camera.isOpened():
			ret, frame = self.camera.read()
			self.threshold = cv2.getTrackbarPos('trh1', 'trackbar')
			frame = self.original(frame)
			
			
			cv2.imshow('original', frame)

			if self.isBgCaptured == 0:
				print('not')
			#  Main operation
			if self.isBgCaptured == 1: 
				img = self.removeBG(frame) 
				img = self.clip(img) 

				frame = self.clip(frame)

				ret, thresh = self.gray_blur_thres(img)
				self.write_video(frame, thresh)

				cv2.imshow('ori', thresh)

				contours, length = self.get_contour(thresh)
				if length > 0:
					maxArea = -1
					for i in range(length):  # find the biggest contour (according to area)
						temp = contours[i]
						area = cv2.contourArea(temp)
						if area > maxArea:
							maxArea = area
							ci = i

					res = contours[ci]
					hull = cv2.convexHull(res)
					drawing = np.zeros(img.shape, np.uint8)
					cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
					cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

					isFinishCal,cnt = self.calculateFingers(res,drawing)
					if self.prev_cnt is not cnt:
						print(cnt)
					self.prev_cnt = cnt
					cv2.imshow('output', drawing)

			# Keyboard OP
			k = cv2.waitKey(10)
			if k == 27:  # press ESC to exit
				break
			elif k == ord('b'):  # press 'b' to capture the background
				self.init_save_video()
				self.bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, bgSubThreshold)
				self.isBgCaptured = 1
				print('!!!Background Captured!!!')
			elif k == ord('r'):  # press 'r' to reset the background
				self.video_num += 1
				self.quit_save_video()
				self.bgModel = None
				self.triggerSwitch = False
				self.isBgCaptured = 0
				print('!!!Reset BackGround!!!')
			elif k == ord('n'):
				self.triggerSwitch = True
				print('!!!Trigger On!!!')
		self.camera.release()


if __name__ == "__main__":
	video_dir = '../data/video/output_real_2.avi'
	FC = FingerCounter(video_dir)
	FC.init_camera()
	FC.init_game()
	FC.play_game()

 






