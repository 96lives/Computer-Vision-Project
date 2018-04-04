import sys
import numpy as np
import cv2
import os 



filename = 'IMG_9504'
extension = '.MOV'
image_size = [640, 480]
cap = cv2.VideoCapture(filename + extension)
frame_num = 0
image_num = 0
while True:
    ret, frame = cap.read()
    if frame_num%30 == 0: #capture every 0.5 seconds
        frame = cv2.resize(frame, (image_size[0], image_size[1]))
        #cv2.imshow('frame',frame)
        cv2.imwrite(filename + "_" + str(image_num) + ".jpg" , frame)
        print(frame_num)
        image_num += 1
    if  ret == False | (cv2.waitKey(1) & 0xFF == ord('q')):
        break
    frame_num += 1

cap.release()
cv2.destroyAllWindows()
