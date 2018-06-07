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

		self.prev_binary = None
		self.smoothed = np.array([])
		self.minima = []
		self.min_image = None
		self.max_image = None

		self.start_frame = 0
		self.num_frame = 0

		self.frame_1 = None
		self.frame_2 = None
		self.frame_3 = None

		#self.bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, 0.5)

	def removeBG(self, frame):
		# apply background subtractor and erode
		fgmask = self.bgModel.apply(frame, learningRate = 0.0000)
		kernel = np.ones((3, 3), np.uint8)
		fgmask = cv2.erode(fgmask, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=fgmask)
		return res

	def local_minmax(self, arr, binary, frame):
		num_min = 0
		num_max = 0
		#cnt = cv2.countNonZero(binary)
		start_amplitude = 10
		amplitude = 12
		mini = 9999
		maxi = -9999
		find = 'start' # find min/max
		arr = np.array(arr)#.reshape((-1,1)) # (a,1) numpy array
		if arr.shape[0] > 3 and self.start_frame is not None:
			arr = np.convolve(arr, [1/16,4/16,6/16,4/16,1/16], 'same')
			arr[0] = arr[2]
			arr[1] = arr[2]
			arr[-1] = arr[-3]
			arr[-2] = arr[-3]
			start = arr[self.start_frame]
			for i in range(max(self.start_frame,0), len(arr)-2): # minimum: slower
				#print(find)
				if find == 'start':
					if (arr[i] > start + start_amplitude) and arr[i-1] > arr[i]:
						num_max += 1
						maxi = arr[i]
						find = 'min'
						#print('max test {} {} {} {} {} '.format(i, arr[i], start, self.start_frame, start_amplitude))
						if self.max_image is None:
							self.max_image = self.frame_3
							print('max saved: {} , frame {}'.format(maxi, i))
					elif (arr[i] < start - start_amplitude) and arr[i-1] < arr[i]:
						num_min += 1
						mini = arr[i]
						find = 'max'
						#print('min test {} {} {} {} {} '.format(i, arr[i], start, self.start_frame, start_amplitude))
						if self.min_image is None:
							self.min_image = self.frame_3
							print('min saved: {} , frame {}'.format(mini, i))
						self.minima.append(arr[i])

				elif find == 'max':
					if (arr[i] > mini + amplitude) and arr[i-1] > arr[i]:
						num_max += 1
						maxi = arr[i]
						find = 'min'
						if self.max_image is None:
							self.max_image = self.frame_3
							print('max saved: {} , frame {}'.format(maxi, i))
					elif (arr[i] < mini - amplitude*2) and arr[i-1] < arr[i]:
						num_min += 1
						mini = arr[i]
						if self.min_image is None:
							self.min_image = self.frame_3
							print('min saved: {} , frame {}'.format(mini, i))
						self.minima.append(arr[i])
				elif find == 'min':
					if (arr[i] < maxi - amplitude) and arr[i-1] < arr[i]:
						num_min += 1
						mini = arr[i]
						find = 'max'
						if self.min_image is None:
							self.min_image = self.frame_3
							print('min saved: {} , frame {}'.format(mini, i))
						self.minima.append(arr[i])
					if (arr[i] > maxi + amplitude*2) and arr[i-1] > arr[i]:
						num_max += 1
						maxi = arr[i]
						if self.max_image is None:
							self.max_image = self.frame_3
							print('max saved: {} , frame {}'.format(maxi, i))
			self.smoothed = arr
			return num_min, num_max
		return 0, 0

	def update(self, binary, frame):
		h = binary.shape[0]
		w = binary.shape[1]
		weighted_y_sum = 0
		weighted_x_sum = 0
		ratio = 0.05

		#res = self.removeBG(frame)
		#res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		#binary = cv2.threshold(res,127,255,cv2.THRESH_BINARY)
		#binary = np.array(binary)

		idx = np.argwhere(binary > 0)
		cnt = len(idx)
		for i in idx:
			weighted_y_sum += i[0]
			weighted_x_sum += i[1]
		if cnt is not 0:
			self.yhistory.append(weighted_y_sum/cnt)
			self.xhistory.append(weighted_x_sum/cnt)
		if cv2.countNonZero(binary) > binary.shape[0]*binary.shape[1]*ratio and self.start_frame is None:
			self.start_frame = self.num_frame
			print("start shake detection : frame "+str(self.num_frame))
		self.num_frame += 1
		self.frame_3 = self.frame_2
		self.frame_2 = self.frame_1
		self.frame_1 = frame


	def get_minmax_image(self):
		if self.min_image is None:
			return False
		if self.max_image is None:
			return False
		cv2.imwrite('maxi.jpg', self.max_image)
		cv2.imwrite('mini.jpg', self.min_image)
		return self.min_image, self.max_image

	def shake_detect(self, binary, frame):
		self.update(binary, frame)
		num_min, num_max = self.local_minmax(self.yhistory, binary, frame)
		if num_min + num_max is not 0: 
			print('local : ' + str(num_min) + ', ' + str(num_max))
		if num_min + num_max >= 4: # and self.yhistory[-1] < max(self.minima) + 15:
			return True
		self.prev_binary = binary
		return False

if __name__ == "__main__":
	sh = Shaker()
	cap = cv2.VideoCapture('output_bin_1.avi')
	while cap.isOpened():
		ret, frame = cap.read()
		frame = cv2.resize(frame,(320, 240))
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
