from shaker import Shaker
import cv2
import skin_detection as sd



class DiffShaker():

    def __init__(self):
        self.shaker = Shaker()

    def shake_detect(self, prev_frame, curr_frame):
        
        blur_size = 11
        threshold = 0.8 * 255
        copy = curr_frame
        curr_frame = cv2.medianBlur(curr_frame, blur_size)
        prev_frame = cv2.medianBlur(prev_frame, blur_size)
        mask = sd.mask_skin(curr_frame)
        diff = abs_diff(curr_frame, prev_frame)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        #diff[diff < threshold] = 0
        diff = cv2.bitwise_and(diff, diff, mask=mask)
        diff = erode_frame(diff)
        cv2.imshow('diff', diff)
        k = cv2.waitKey(5) & 0xFF
        
        return self.shaker.shake_detect(diff, copy) 

    def get_minmax_image(self):
        out = self.shaker.get_minmax_image()
        self.yhistory = self.shaker.yhistory
        self.smoothed = self.shaker.smoothed
        return out


def abs_diff(f1, f2):
    return abs(f1 - f2)

def erode_frame(mask):
    #kernel_ellipse7= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kernel_ellipse5= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #kernel_ellipse3= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    out = cv2.erode(mask,kernel_ellipse5,iterations = 1)    
    out = cv2.medianBlur(out,7)
    return out
 


