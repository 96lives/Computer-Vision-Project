import cv2
import numpy as np

class visualizer():
    def __init__(self):
        imdir = "../images/"
        self.rps = cv2.imread(imdir + "rps.png")
        self.r = self.gray(self.rps, 0, 106)
        self.s = self.gray(self.rps, 106, 201)
        self.p = self.gray(self.rps, 201, 320)

    def gray(self, rps, min, max):
        left = rps[:,0:min,:]
        middle = rps[:,min:max,:]
        right = rps[:,max:320,:]

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        left_gray = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB)
        right_gray = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)

        im = ()
        if left_gray is not None: 
            im += (left_gray,)
        if middle is not None: 
            im += (middle,)
        if right_gray is not None: 
            im += (right_gray,)

        return np.concatenate(im, axis = 1)

    def show_rps(self, image, rps):
        if rps == 'r':
            image = np.concatenate((image, self.r/255.0), axis = 0)
        elif rps == 's':
            image = np.concatenate((image, self.s/255.0), axis = 0)
        elif rps == 'p':
            image = np.concatenate((image, self.p/255.0), axis = 0)
        cv2.imshow('image', img)
        cv2.waitKey(0)
    
if __name__ == '__main__':
    v = visualizer()
    im = np.random.rand(240, 320, 3)
    img = v.show_rps(im, 'r')
    cv2.imshow('image', img)
    cv2.waitKey(10000)

