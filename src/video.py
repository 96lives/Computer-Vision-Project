import os 
import fnmatch
import skin_detection as sd
from finger_counter2 import FingerCounter2
import cv2
import matplotlib.pyplot as plt

#file_name = "test.MOV"
#out_dir = "output.avi"
#cap = cv2.VideoCapture(file_name)
#fc = FC.FingerCounter('skin', file_name, out_dir)
#fc.play_game()
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

if __name__ == "__main__":
    data_dir = "../data/"
    out_dir = data_dir + "out/"
    folders = ['P/', 'R/', 'S/']
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
                    fc = FingerCounter2(f, \
                            "report.txt", in_dir, \
                            out_dir + folder)
                    cnt_list = fc.play_game()
                    if len(cnt_list) != 0:
                        print(cnt_list)
                        print(len(cnt_list))
                        plt.plot(cnt_list)
                    cnt += 1    
                    total_cnt += 1
        plt.savefig(subdir+"finger_cnts.png")
        plt.clf()

    #print("activated number of shaker: " + str(activated_cnt))
    #print("Whole dataset: " + str(total_cnt))



