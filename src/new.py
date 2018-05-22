import cv2
import numpy as np
import copy
import math	
from gtts import gTTS
import os
import threading
import sys
RED   = "\033[1;31m"
RESET = "\033[0;0m"


# Environment:
# OS    : Ubuntu OS 17.10
# python: 3.6.3
# opencv: 3.4.0

# pip install gtts
# sudo apt-get install mpg321

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    #TODO: parameter(learning rate) tuning
    fgmask = bgModel.apply(frame, learningRate = 0.0000)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

def speak(speech, filename = 'audio', slow = False):
    print('\033[1;32;40m [TTS] speak : ' + speech + '\033[0m')
    tts = gTTS(speech, slow = slow)
    tts.save(filename+'.mp3')
    os.system('mpg321 '+filename+'.mp3')

global speak_end
speak_end = False    

def speak_rps():
    speak('Rock', 'rock', True)
    speak('Scissors', 'scissors', True)
    speak('Paper!', 'paper', False)
    global speak_end
    speak_end = True

def speak_rps2():
    speak('Rock, Paper, Scissors!', 'rps', True)
    global speak_end
    speak_end = True

def speak_rps_thread():
    t = threading.Thread(target = speak_rps2)
    t.start()

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,100)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

#speak('Hello, Let\'s play rock scissors paper! Please press b to capture background.', 'start', False)

#fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
#fourcc = cv2.VideoWriter_fourcc(*'H264') 
#fourcc = cv2.VideoWriter_fourcc(*'X264') 
#fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
#fourcc = cv2.VideoWriter_fourcc(*'DIVX') 
#fourcc = cv2.VideoWriter_fourcc(*'WMV1')


with open('../data/video/label.txt','r') as f:
    num = int(f.readline())

while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        sys.exit()
    elif k == ord('b'):  # press 'b' to capture the background

        writer_bin = cv2.VideoWriter('../../data/video/output_bin_'+str(num)+'.avi',fourcc, 20.0, (320,384))
        writer_real = cv2.VideoWriter('../../data/video/output_real_'+str(num)+'.avi',fourcc, 20.0, (320,384))

        bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
        break

#speak_rps_thread()
prev_cnt = -1





while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        #cv2.imshow('mask', img)

        frame = frame[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]] 
        writer_real.write(frame)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        #cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        towrite = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)

        writer_bin.write(towrite)

        cv2.imshow('ori', thresh)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            if prev_cnt is not cnt:
                print(cnt)
            prev_cnt = cnt
        cv2.imshow('output', drawing)
        if speak_end is True:
            print('===Play a throw===')
            if cnt == 0:
	        #paper = cv2.imread('paper.jpg')
	        #cv2.imshow('result', paper)
                print('paper')
            elif cnt == 1:
	        #rock = cv2.imread('rock.jpg')
                #cv2.imshow('result', rock)
                print('rock')
            elif cnt >= 2:
                #scissors = cv2.imread('scissors.jpg')
                #cv2.imshow('result', scissors)
                print('scissors')
            ret = input("Press return to continue playing game: ")
            speak_end = False
            speak_rps_thread()
            continue


    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        writer_bin = cv2.VideoWriter('../../data/video/output_bin_'+str(num)+'.avi',fourcc, 20.0, (320,384))
        writer_real = cv2.VideoWriter('../../data/video/output_real_'+str(num)+'.avi',fourcc, 20.0, (320,384))
        bgModel = cv2.createBackgroundSubtractorMOG2(2147483647, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        writer_bin.release()
        writer_real.release()
        num += 1
        with open('../../data/video/label.txt','w') as f:
            f.write(str(num))
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('!!!Reset BackGround!!!')
    elif k == ord('n'):
        triggerSwitch = True
        print('!!!Trigger On!!!')
