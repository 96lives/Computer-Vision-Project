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
				qualityLevel = 0.1,
				minDistance = 7,
				blockSize = 7 )

		# Parameters for lucas kanade optical flow
		self.lk_params = dict(winSize = (15,15),
				maxLevel = 2,
				criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

		self.prev_binary = None
		self.smoothed = np.array([])
		self.minima = []
		self.min_bianry = None
		self.max_bianry = None
		self.min_image = None
		self.max_image = None

	def optical_flow(self, binary):
		self.flow_pix = []
		if self.prev_binary is None or binary is None:
			return None, None, None
		try:
			p0 = cv2.goodFeaturesToTrack(self.prev_binary, mask = None, **self.feature_params)
			p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_binary, binary, p0, None, **self.lk_params)
		except Exception as e:
			self.prev_binary = binary.copy()
			return None, None, None
		return p0, p1, st

	def update_flow(self, p0, p1, st):
		good_new = p1[st==1]
		good_old = p0[st==1]
		#x_cor = []
		#y_cor = []
		flow_thres = 15
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			#x_cor.append(a)
			#y_cor.append(b)
			if math.sqrt((a-c)**2+(b-d)**2) > flow_thres:
				self.flow_pix.append([a, b])
		#if x_cor is not []:
		#	self.xhistory.append(np.average(x_cor))
		#if y_cor is not []:
		#	self.yhistory.append(np.average(y_cor))
		#print('update (' + str(np.average(x_cor)) + ', ' + str(np.average(y_cor))+')')

	def local_minmax(self, arr, binary, frame):
		num_min = 0
		num_max = 0
		margin = 2
		arr = np.array(arr)#.reshape((-1,1)) # (a,1) numpy array
		if arr.shape[0] > 20:
			#arr = gaussian_filter(arr, sigma=7)
			arr = np.convolve(arr, [1/16,4/16,6/16,4/16,1/16], 'same')
			arr[0] = arr[2]
			arr[1] = arr[2]
			arr[-1] = arr[-3]
			arr[-2] = arr[-3]
			for i in range(20,len(arr)-1): # minimum: slower
				if (arr[i] < arr[i-1] - margin and arr[i] < arr[i+1]) \
					or (arr[i] < arr[i-1] and arr[i] < arr[i+1] - margin) :
					num_min += 1
					if num_min is 1:
						self.min_image = frame
					self.minima.append(arr[i])
				if (arr[i] > arr[i-1] + margin and arr[i] > arr[i+1]) \
					or (arr[i] > arr[i-1] and arr[i] > arr[i+1] + margin) :
					num_max += 1
					if num_max is 1:
						self.max_image = frame
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
		k = cv2.waitKey(10)
		if k == 27:
			pass

	def update2(self, binary):
		h = binary.shape[0]
		w = binary.shape[1]
		weighted_y_sum = 0
		weighted_x_sum = 0
		idx = np.argwhere(binary > 0)
		cnt = len(idx)
		for i in idx:
			weighted_y_sum += i[0]
			weighted_x_sum += i[1]
		if cnt is not 0:
			self.yhistory.append(weighted_y_sum/cnt)
			self.xhistory.append(weighted_x_sum/cnt)

	def get_flow_pixel(self):
		return self.flow_pix

	def get_minmax_image(self):
		if self.min_image is None:
			return False
		if self.max_image is None:
			return False
		return self.min_image, self.max_image

	def shake_detect(self, binary, frame):
		#p0, p1, st = self.optical_flow(binary)
		#if p0 is not None:
		#	self.update_flow(p0, p1, st)
		self.update2(binary)
		num_min, num_max = self.local_minmax(self.yhistory, binary, frame)
		print('local : ' + str(num_min) + ', ' + str(num_max))
		if num_min >= 1 and num_max >= 2 and self.yhistory[-1] < max(self.minima) + 30:
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
