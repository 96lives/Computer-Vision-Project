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
    data_dir = "../data/"
    out_dir = data_dir + "plot/"
    folders = ['hardP/', 'hardR/', 'hardS/']
    pattern = "*.MOV"
    
    total_cnt = 0
    activated_cnt = 0

    for folder in folders:
        subdir = data_dir + folder
        if os.path.exists(subdir):
            cnt = 0 
            for f in files(subdir):
                if fnmatch.fnmatch(f, pattern):
                    in_dir = subdir
                    file_name = folder + 'out' + f
                    fc = NewFingerCounter(f, \
                            "report.txt", in_dir, \
                            out_dir)
                    cnt_list = fc.play_game()
                    #time.sleep(2)
                    cnt += 1    
                    total_cnt += 1
        plt.savefig(subdir+"finger_cnts.png")
        plt.clf()

    #print("activated number of shaker: " + str(activated_cnt))
    #print("Whole dataset: " + str(total_cnt))



