import os 
import fnmatch
import skin_detection as sd
from finger_counter_tester import FingerCounterTester
import cv2

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
    out_dir = data_dir + "plot/"
    folders = ['P/', 'R/', 'S/']
    pattern = "*.MOV"

    for folder in folders:
        subdir = data_dir + folder
        if os.path.exists(subdir):
            for f in files(subdir):
                if fnmatch.fnmatch(f, pattern):
                    in_dir = subdir
                    file_name = folder + 'out' + f
                    fc = FingerCounterTester(f, \
                            "report.txt", in_dir, \
                            out_dir)
                    fc.play_game()
                
