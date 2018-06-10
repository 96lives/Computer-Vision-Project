import cv2
import os 
import skin_detection as sd
from random import *

i = 1
cam = cv2.VideoCapture(0)
BIG_NUM = 1000000000

while True:
    ret_val, img = cam.read()
    frame_size = (240, 180)
    img = cv2.resize(img, frame_size)
    img = sd.mask_skin(img)
    cv2.imshow('my webcam', img)
    k = cv2.waitKey(1)
    out_dir = "../data/train-data/"
    folder = out_dir + "R/"

    if k==27:
        break
    elif k==49:
        i = 1
        print("Changed to R")
        folder = out_dir + "R/"

    elif k==50:
        print("Changed to P")
        folder = out_dir + "P/"

    elif k==51:
        print("Changed to S")
        folder = out_dir + "S/"

    elif k==32:
        write_name = folder + \
                str(randint(1, BIG_NUM)) + '.png'
        cv2.imwrite(write_name, img)
        print("Saved ", write_name)
cv2.destroyAllWindows()
