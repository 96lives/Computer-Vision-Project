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
        pos, neg = self.collect_data(min_img, max_img)
        #self.classifier = rf.RFClassifier(pos, neg)
        self.classifier = knn.KNNClassifier(pos, neg)
     
    def collect_data(self, min_img, max_img):
        
        min_mask = sd.mask_skin(min_img)
        max_mask = sd.mask_skin(max_img)

        and_mask = cv2.bitwise_and(min_mask, max_mask)
        
        # mask neg image of min_image
        neg = self.collect_neg_data(min_img, and_mask)

        # mask pos image of min_image
        pos = self.collect_pos_data(min_img, min_mask, and_mask)
        # mask pos image of max_image
        #pos = np.append(pos, \
        #    self.collect_pos_data(max_img, max_mask, and_mask), axis=0)

        return pos, neg


    def collect_neg_data(self, img, and_mask):
        
        img = cv2.bitwise_and(img, img, mask=and_mask)
        and_mask = and_mask.reshape(-1, 1)
        img = img.reshape(-1, 3)
        neg_bool = np.nonzero(and_mask)
        return img[neg_bool[0]]

    def collect_pos_data(self, img, img_mask, and_mask):

        pos_mask = img_mask - and_mask
        pos_mask[pos_mask < 0] = 0
        pos_img = cv2.bitwise_and(img, img, mask=pos_mask)
        img = pos_img.reshape(-1, 3)
        pos_mask = pos_mask.reshape(-1, 1)
        pos_bool = np.nonzero(pos_mask)
        return img[pos_bool[0]]    
      
    def mask_with_classifier(self, img):

        #blur_k_size = (11, 11)        
        #img = cv2.blur(img, blur_k_size)
        w, h, c = img.shape
        skin_mask = sd.mask_skin(img)
       
        # determine candidates for classification using mask
        skin_mask_reshaped = skin_mask.reshape(-1, 1)
        candidates = (np.nonzero(skin_mask_reshaped))[0]

        # reshaped image for collecting pixels
        reshaped_img = img.reshape(-1, 3)
        out = np.zeros(reshaped_img.shape)

        # classify
        logits =  self.classifier.classify(reshaped_img[candidates])

        # collect skin pixels
        skin_idx = np.nonzero(np.array(logits) == 1)
        out[candidates[skin_idx]] = 255
        out = out.reshape(w, h, c)
        return out

if __name__ == '__main__':


    # test data
    dir1 = "../test-data/img_min.png"
    dir2 = "../test-data/img_max.png"
    test_dir = "../test-data/img_test.png"

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
    
