import RFClassifier as rf
import skin_detection as sd
import numpy as np
import time

class SkinColorClassifier():

    # min_img, max_img is non masked image
    # both image ensures that hand image exists 
    # and location of the hand image is different
    def __init__(self, min_img, max_img):

        # mask both images and classify 
        self.mask1 = sd.mask_skin(frame)
        self.mask2 = sd.mask_skin(frame)
        
        # train classifier
        pass

    # gets image and returns masked image
    def mask_with_classifier(self.img):
        pass
        