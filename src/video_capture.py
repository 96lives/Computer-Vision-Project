import os 
import fnmatch
import skin_detection as sd
from finger_counter import FingerCounter
from new_finger_counter import NewFingerCounter
import cv2
import matplotlib.pyplot as plt
import time

def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

if __name__ == "__main__":
    data_dir = "../data/train-data/"
    out_dir = "capture/"
    folders = ['trainR/', 'trainS/']
    pattern = "*.avi"
    
    total_cnt = 0
    activated_cnt = 0

    for folder in folders:
        sub_dir = data_dir + folder
        if os.path.exists(sub_dir):
            for f in files(sub_dir):
                if fnmatch.fnmatch(f, pattern):
                    print(sub_dir + f)
                    real_name = f.replace('.avi', '')
                    cap = cv2.VideoCapture(sub_dir+f)
                    frame_cnt = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if frame_cnt > 45 and frame_cnt % 2 == 0:
                            out_name = real_name + str(frame_cnt) +'.jpg'
                            out_full = sub_dir + out_dir + out_name
                            print(out_full)
                            cv2.imwrite(out_full, frame)
                        frame_cnt += 1
                    cap.release()




