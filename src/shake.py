import cv2
import numpy as np
import copy
import math	
import os
import sys
import matplotlib.pyplot as plt
import time
#from scipy.ndimage.filters import gaussian_filter

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
		self.smoothed = np.array([])
		self.minima = []

	def optical_flow(self, binary):
		if self.prev_binary is None or binary is None:
			return None, None, None
		p0 = cv2.goodFeaturesToTrack(self.prev_binary, mask = None, **self.feature_params)
		try:
			p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_binary, binary, p0, None, **self.lk_params)
		except Exception as e:
			self.prev_binary = binary.copy()
			print('yes')
			return None, None, None
		return p0, p1, st

	def update(self, p0, p1, st):
		good_new = p1[st==1]
		good_old = p0[st==1]
		x_cor = []
		y_cor = []
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			x_cor.append(a)
			y_cor.append(b)
		self.xhistory.append(np.average(x_cor))
		self.yhistory.append(np.average(y_cor))
		#print('update (' + str(np.average(x_cor)) + ', ' + str(np.average(y_cor))+')')

	def local_minmax(self, arr):
		num_min = 0
		num_max = 0
		arr = np.array(arr)#.reshape((-1,1)) # (a,1) numpy array
		if arr.shape[0] > 10: 
			#arr = gaussian_filter(arr, sigma=7)
			arr = np.convolve(arr, [1/16,4/16,6/16,4/16,1/16], 'same')
			arr[0] = arr[1]
			arr[-1] = arr[-2]
			for i in range(1,len(arr)-1):
				if arr[i] < arr[i-1] and arr[i] < arr[i+1] :
					num_min += 1
					self.minima.append(arr[i])
				if arr[i] > arr[i-1] and arr[i] > arr[i+1] :
					num_max += 1
			self.smoothed = arr
			return num_min, num_max
		return 0, 0



	def visualize(self, binary, p0, p1, st):
		mask = np.zeros_like(binary)
		good_new = p1[st==1]
		good_old = p0[st==1]
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
			frame = cv2.circle(binary,(a,b),5,color[i].tolist(),-1)
		img = cv2.add(frame,mask)
		cv2.imshow('frame',img)
		k = cv2.waitKey(10)
		if k == 27:
			pass

	def shake_detect(self, binary):
		p0, p1, st = self.optical_flow(binary)
		if p0 is None:
			self.prev_binary = binary
			return False
		self.update(p0, p1, st)
		num_min, num_max = self.local_minmax(self.xhistory)
		print('local : ' + str(num_min) + ', ' + str(num_max))
		if num_min >= 1 and num_max >= 2 and self.xhistory[-1] < self.minima[0] + 30:
			return True
		self.prev_binary = binary
		return False
		
if __name__ == "__main__":
	sh = Shaker()
	cap = cv2.VideoCapture('output_bin_1.avi')
	while cap.isOpened():
		ret, frame = cap.read()
		frame = cv2.resize(frame,(640, 480))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if ret is False: 
			break
		cv2.imshow('original', frame)

		ret = sh.shake_detect(frame)
		if ret is True: 
			break
		k = cv2.waitKey(10)
		if k == 27:
			break
		

	plt.plot(sh.xhistory)
	plt.ylabel('avg x')
	#plt.show()

	plt.plot(sh.smoothed)
	plt.ylabel('smoothed')
	plt.show()

 






