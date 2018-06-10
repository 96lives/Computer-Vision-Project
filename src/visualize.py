import cv2
import numpy as np

class visualizer():
    def __init__(self):
        imdir = "../images/"
        self.rps = cv2.imread(imdir + "rps.png")
        self.r = self.gray(self.rps, 0, 106)
        self.s = self.gray(self.rps, 106, 201)
        self.p = self.gray(self.rps, 201, 320)

        self.cnt_list = []
        self.mu = 0
        self.rps = 'r'

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
        ratio = image.shape[1] / self.r.shape[1]
        if rps == 'r':
            rpsimg = cv2.resize(self.r, (image.shape[1], round(self.r.shape[0]*ratio)))
        elif rps == 's':
            rpsimg = cv2.resize(self.s, (image.shape[1], round(self.s.shape[0]*ratio)))
        elif rps == 'p':
            rpsimg = cv2.resize(self.p, (image.shape[1], round(self.p.shape[0]*ratio)))
        image = np.concatenate((image, rpsimg), axis = 0)
        return image

    def visualize(self, image, finger_cnt, decision_cnt, refresh = False):
        alpha = 0.3
        skip_frames = 8
        if decision_cnt > skip_frames:
            self.mu = alpha * finger_cnt + (1-alpha) * self.mu
            self.cnt_list.append(self.mu)
            if not refresh:
                if self.mu > 1.9 and self.rps in ['r','s']:
                    self.rps = 'p'
                elif self.mu > 0.9 and self.rps is 'r':
                    self.rps = 's'
            else:
                if self.mu > 1.9:
                    self.rps = 'p'
                elif self.mu > 0.9:
                    self.rps = 's'
                else:
                    self.rps = 'r'
        return self.show_rps(image, self.rps)
    
if __name__ == '__main__':
    v = visualizer()
    im = np.random.rand(240, 320, 3)
    img = v.show_rps(im, 'r')
    cv2.imshow('image', img)
    cv2.waitKey(1000)

