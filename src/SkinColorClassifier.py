import RFClassifier as rf
import KNNClassifier as knn
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
        #self.classifier = rf.RFClassifier(pos, neg)
        self.classifier = knn.KNNClassifier(pos, neg)
     
    def get_data(self, min_img, max_img):
        
        min_mask = sd.mask_skin(min_img)
        max_mask = sd.mask_skin(max_img)

        and_mask = cv2.bitwise_and(min_mask, max_mask)
        
        start_time = time.time()
        # mask neg image of min_image
        neg_img = cv2.bitwise_and(min_img, min_img, mask=and_mask)
        reshaped_and_mask = and_mask.reshape(-1, 1)
        neg_img = neg_img.reshape(-1, 3)
        neg_bool = np.nonzero(reshaped_and_mask)
        neg = neg_img[neg_bool[0]]

        # mask pos image of min_image
        pos_mask = min_mask - and_mask
        pos_mask[pos_mask < 0] = 0
        pos_img = cv2.bitwise_and(min_img, min_img, mask=pos_mask)
        pos_img = pos_img.reshape(-1, 3)
        reshaped_pos_mask = pos_mask.reshape(-1, 1)
        pos_bool = np.nonzero(reshaped_pos_mask)
        pos = pos_img[pos_bool[0]]    
        #print("add count in min image: " + str(pos.shape))

        print("Image processing time: " + str(time.time() - start_time))
        # mask pos image of max_image
        pos_mask2 = max_mask - and_mask
        pos_mask2[pos_mask2 < 0] = 0
        pos_img2 = cv2.bitwise_and(max_img, max_img, mask=pos_mask2)
        pos_img2 = pos_img2.reshape(-1, 3)
        reshaped_pos_mask2 = pos_mask2.reshape(-1, 1)
        pos_bool2 = np.nonzero(reshaped_pos_mask2)
        pos = np.append(pos, pos_img2[pos_bool2[0]], axis=0)
        #print("add count in min and max image: " + str(pos.shape))

        return pos, neg
    
    
    def mask_with_classifier(self, img):

        #blur_k_size = (11, 11)        
        #img = cv2.blur(img, blur_k_size)
        
        w, h, c = img.shape
        mask = sd.mask_skin(img)
       
        reshaped_mask = mask.reshape(-1, 1)
        candidates = (np.nonzero(reshaped_mask))[0]

        reshaped_img = img.reshape(-1, 3)

        logits =  self.classifier.classify(reshaped_img[candidates])
        skin_idx = np.nonzero(np.array(logits) == 1)
        reshaped_img[candidates[skin_idx]] = [0, 0, 255]
        out = reshaped_img.reshape(w, h, c)
        return out

if __name__ == '__main__':

    dir1 = "../data/img_min.png"
    dir2 = "../data/img_max.png"
    test_dir = "../data/img_test.png"

    min_img = cv2.imread(dir1)
    min_img = cv2.resize(min_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    max_img = cv2.imread(dir2)
    max_img = cv2.resize(max_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    test_img = cv2.imread(test_dir)
    test_img = cv2.resize(test_img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    copy = test_img 
   
    start_time = time.time()
    scc = SkinColorClassifier(min_img, max_img)
    out = scc.mask_with_classifier(test_img)
    print("Time duration: " + str(time.time() - start_time))
    
    cv2.imshow('fixed mask', sd.mask_skin(copy))
    cv2.imshow('rf mask', out)
    cv2.waitKey(0)
    
