import cv2
import numpy as np
import copy
import math	
import os
import sys

class Shaker():
	def __init__(self):
		self.xhistory = []
		self.yhistory = []
		self.count = 0

		# params for ShiTomasi corner detection
		self.feature_params = dict(maxCorners = 100,
				qualityLevel = 0.3,
				minDistance = 7,
				blockSize = 7 )

		# Parameters for lucas kanade optical flow
		self.lk_params = dict(winSize = (15,15),
				maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		self.prev_binary = None
		self.binary = None

	def optical_flow(self, binary):
		if binary is None:
			return None, None, None
		p0 = cv2.goodFeaturesToTrack(self.prev_binary, mask = None, **feature_params)
		try:
        		p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_binary, binary, p0, None, **lk_params)
		except:
			self.prev_binary = binary.copy()
			return None, None, None
		return p0, p1, st

	def update(self, p0, p1, st):
		good_new = p1[st==1]
		good_old = p0[st==1]
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			self.xhistory.append(a)
			self.yhistory.append(b)

	def local_minmax(self, arr):
		num_min = 0
		num_max = 0
		for i in len(arr):
			if arr[i] < arr[i-1] and arr[i] < arr[i+1] and i > 5:
				num_min += 1
			if arr[i] > arr[i-1] and arr[i] > arr[i+1] and i > 5:
				num_max += 1
		return num_min, num_max

	def shake_detect(self, binary):
		p0, p1, st = self.optical_flow(binary)
		if p0 is None:
			return False
		self.update(p0, p1, st)
		
		


if __name__ == "__main__":
	FC = FingerCounter()
	FC.init_camera()
	FC.init_game()
	FC.play_game()


	# FingerCounter(video_dir, save_dir)
	data_dir = 'data/'
	save_dir = 'data/Gen/'
	filenames = os.listdir(data_dir)
	for f in filenames:
		FC = FingerCounter(video_dir = data_dir + f, save_dir = save_dir)
		FC.init_camera()
		FC.init_game()
		FC.play_game()

 






