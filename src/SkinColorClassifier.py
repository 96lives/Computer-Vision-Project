import RFClassifier as rf
import skin_detection as sd
import numpy as np
import cv2
import time

class SkinColorClassifier():

    # min_img, max_img is non masked image
    # both image ensures that hand image exists 
    # and location of the hand image is different
    def __init__(self, min_img, max_img):

        # mask both images and classify
        pos, neg = self.get_data(min_img, max_img)
        self.rf = rf.RFClassifier(pos, neg)
        
     
    def get_data(self, min_img, max_img):

        # get and and  
        min_mask = sd.mask_skin(min_img)
        max_mask = sd.mask_skin(max_img)

        and_mask = cv2.bitwise_and(min_mask, max_mask)
        
        neg_img = cv2.bitwise_and(min_img, min_img, mask=and_mask)
        # maybe add neg2 = cv2.bitwise_and(max_img, max_img, mask=and_mask)
        reshaped_and_mask = and_mask.reshape(-1, 1)
        neg_img = neg_img.reshape(-1, 3)
        neg_bool = np.nonzero(reshaped_and_mask)
        neg = neg_img[neg_bool[0]]
     
        pos_mask = min_mask - and_mask
        pos_mask[pos_mask < 0] = 0
        pos_img = cv2.bitwise_and(min_img, min_img, mask=pos_mask)
        pos_img = pos_img.reshape(-1, 3)
        reshaped_pos_mask = pos_mask.reshape(-1, 1)
        pos_bool = np.nonzero(reshaped_pos_mask)
        pos = pos_img[pos_bool[0]]    

        return pos, neg
    
    def mask_with_classifier(self, img):
        
        w, h, c = img.shape
        mask = sd.mask_skin(img)
        
        reshaped_mask = mask.reshape(-1, 1)
        candidates = (np.nonzero(reshaped_mask))[0]
        print(candidates)

        reshaped_img = img.reshape(-1, 3)
        #reshaped_img[candidates] = [255, 0,0 ]
        #reshaped_img = reshaped_img.reshape(w, h, c)
        #cv2.imshow('z',reshaped_img)
        #cv2.waitKey(0)

        logits =  self.rf.classify(reshaped_img[candidates])
        print(logits)
        skin_idx = np.nonzero(np.array(logits) == 1)
        reshaped_img[candidates[skin_idx]] = [0, 0, 255]
        
        return reshaped_img.reshape(w, h, c) 

if __name__ == '__main__':

    dir1 = "../data/img_min.png"
    dir2 = "../data/img_max.png"
    test_dir = "../data/img_test.png"
    min_img = cv2.imread(dir1)
    min_img = cv2.resize(min_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    max_img = cv2.imread(dir2)
    max_img = cv2.resize(max_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    test_img = cv2.imread(test_dir)
    copy = test_img 
    scc = SkinColorClassifier(min_img, max_img)
    out = scc.mask_with_classifier(test_img)
    cv2.imshow('fixed mask', sd.mask_skin())
    cv2.imshow('rf mask', out)
    cv2.waitKey(0)
    
