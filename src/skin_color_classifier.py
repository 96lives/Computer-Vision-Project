import rf_classifier as rf
import knn_classifier as knn
import skin_detection as sd
import numpy as np
import cv2
import time


out_dir = "../data/out/"


class SkinColorClassifier():

    # min_img, max_img is non masked image
    # both image ensures that hand image exists 
    # and location of the hand image is different
    def __init__(self, img1, img2, classifier="knn"):

        # mask both images and classify 
        pos, neg = self.collect_data(img1, img2)
        if classifier == "knn":
            self.classifier = knn.KNNClassifier(pos, neg)
        elif classifier == "rf":
            self.classifier = rf.RFClassifier(pos, neg)
        else:
            print("classifier argument not allowed")

    def collect_data(self, img1, img2):
        
        mask1 = sd.mask_skin(img1)
        mask2 = sd.mask_skin(img2)

        and_mask = cv2.bitwise_and(mask1, mask2)
        
        # mask neg image of min_image
        neg = self.collect_neg_data(img1, and_mask)

        # mask pos image of min_image
        pos = self.collect_pos_data(img1, mask1, and_mask)
        # mask pos image of max_image
        pos = np.append(pos, \
            self.collect_pos_data(img2, \
            mask2, and_mask), axis=0)

        return pos, neg


    def collect_neg_data(self, img, and_mask):
        
        img = cv2.bitwise_and(img, img, mask=and_mask)
        #cv2.imwrite(out_dir + "neg_data.jpg", img)
        and_mask = and_mask.reshape(-1, 1)
        img = img.reshape(-1, 3)
        neg_bool = np.nonzero(and_mask)
        return img[neg_bool[0]]

    def collect_pos_data(self, img, img_mask, and_mask):

        pos_mask = img_mask - and_mask
        pos_mask[pos_mask < 0] = 0
        pos_img = cv2.bitwise_and(img, img, mask=pos_mask)
        #cv2.imwrite(out_dir + "pos_data.jpg", pos_img)
        img = pos_img.reshape(-1, 3)
        pos_mask = pos_mask.reshape(-1, 1)
        pos_bool = np.nonzero(pos_mask)
        return img[pos_bool[0]]    
      
    def mask_image(self, img):

        #blur_k_size = (11, 11)        
        #img = cv2.blur(img, blur_k_size)
        w, h, c = img.shape
        skin_mask = sd.mask_skin(img)
       
        # determine candidates for classification using mask
        skin_mask_reshaped = skin_mask.reshape(-1, 1)
        candidates = (np.nonzero(skin_mask_reshaped))[0]

        # reshaped image for collecting pixels
        reshaped_img = img.reshape(-1, 3)
        out = np.zeros(w*h, np.uint8)

        # classify
        start_time = time.time()
        logits =  self.classifier.classify(reshaped_img[candidates])
        # collect skin pixels
        skin_idx = np.nonzero(np.array(logits) == 1)
        out[candidates[skin_idx]] = 255
        out = out.reshape(w, h)
        kernel= cv2.getStructuringElement(\
                cv2.MORPH_ELLIPSE,(5,5))
        out = cv2.erode(out, kernel, iterations = 1)    
        return out

if __name__ == '__main__':

    # test data 
    folder_dir = "../test-data/"
    img1_dir = folder_dir + "img_max.png"
    img2_dir = folder_dir + "img_min.png"
    test_dir = folder_dir + "img_test.png"
    '''
    img1_dir = "../test-data/classifier_test1.png"
    img2_dir = "../test-data/classifier_test2.png"
    test_dir = "../test-data/classifier_test3.png"
    '''
    min_img = cv2.imread(img1_dir)
    min_img = cv2.resize(min_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    max_img = cv2.imread(img2_dir)
    max_img = cv2.resize(max_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    test_img = cv2.imread(test_dir)
    test_img = cv2.resize(test_img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    copy = test_img 
   
    start_time = time.time()
    scc = SkinColorClassifier(min_img, max_img)
    out = scc.mask_image(test_img)
    print("Whole Time duration: " + str(time.time() - start_time))
    print("Test Image Size: " + str(test_img.shape))

    cv2.imshow('fixed mask', sd.mask_skin(copy))
    cv2.imshow('rf mask', out)
    cv2.waitKey(0)
    
