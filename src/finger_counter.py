import cv2
import math
import skin_detection as sd
import background_subtractor as bg
import shaker as sh
import matplotlib.pyplot as plt
import time
from skin_color_classifier import SkinColorClassifier


class FingerCounter():

    def __init__(self, mode, \
            in_dir=None, out_dir=None):
        
        if mode == 'background':
            self.is_background = True
        elif mode == 'skin':
            self.is_background = False
        else:
            raise UnavailableModeError()

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.is_webcam = True
        if in_dir is not None:
            self.is_webcam = False
        
    def play_game(self):
        cap = None
        bgs = None

        shaker = sh.Shaker()
        shake_switch = False
        shake_ended = False
        scc = None

        if self.is_background:
            bgs = bg.BackgroundSubtractor(self.is_webcam)

        if self.is_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.in_dir)
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(self.out_dir, fourcc,\
                    round(cap.get(5)), \
                    (int(cap.get(3)),int(cap.get(4))))
       
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            frame = cv2.resize(frame,(320,240))

            if not self.is_webcam:
                frame = cv2.flip(frame, 0)
                frame = cv2.flip(frame, 1)

            if self.is_background:
                mask = bgs.process_frame(frame)
                if mask is None:
                    continue
            else:
                mask = sd.detect_skin(frame)
            cv2.imshow('mask', mask)

            if shake_ended is True:
                if shake_switch is False:
                    print('shake ended')
                    time.sleep(2)
                    shake_switch = True
                    img1, img2 = shaker.get_minmax_image()
                    scc = SkinColorClassifier(img1, img2)

                frame, finger_cnt = count_finger(frame, mask)
                print(finger_cnt)
    
            if shake_switch is False:
                shake_ended = shaker.shake_detect(mask, frame)
            else:
                mask1 = scc.mask_image(frame)
                cv2.imshow('scc mask', mask1)

            cv2.imshow('frame', frame)
            #time.sleep(0.5)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
        
        plt.plot(shaker.yhistory)
        plt.ylabel('avg y')
        #plt.show()
        
        plt.plot(shaker.smoothed)
        plt.ylabel('smoothed')
        plt.savefig('plot.png')
        plt.show()
        cap.release()
        cv2.destroyAllWindows()

class UnavailableModeError(Exception):
    
    def __str__(self):
        return "only 'skin' or 'background' is available"

def count_finger(frame, mask):
    if mask is None:
        return frame, 0
    max_contour = find_max_contour(mask)
    if max_contour is None: 
        return frame, 0

    x,y,w,h = cv2.boundingRect(max_contour)
    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    hull = cv2.convexHull(max_contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(max_contour, hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                a = compute_distance(end, start)
                b = compute_distance(far, start)
                c = compute_distance(end, far)
                angle = compute_angle(a, b, c)
                # treat fingers with angle <= 90
                if angle <= (math.pi / 2):  
                    cnt += 1
                    frame = cv2.circle(frame, \
                            far, 8, [211, 84, 0], -1)
            return frame, cnt
    return frame, 0

def find_max_contour(mask):
 
    _, contours, hierarchy = cv2.findContours(mask,\
            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Find Max contour area (Assume that hand is in the frame)
    max_area=400
    ci=0
    contour = -1
    if contours is None:
        return None
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if(area>max_area):
            max_area=area
            ci=i

    #Largest area contour
    if (len(contours) != 0):
        max_contour = contours[ci]
        return max_contour
    else:
        return None

# computes the L2 distance of tuple a and b
def compute_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 \
            + (a[1] - b[1]) ** 2)

def compute_angle(a, b, c):
    eps = 1e-10
    return math.acos((b**2 + c**2 - a**2) \
        / (3*b*c + eps))
