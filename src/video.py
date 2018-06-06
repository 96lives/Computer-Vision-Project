import os 
import fnmatch
import skin_detection as sd
from finger_counter import FingerCounter
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
    output_dir = data_dir + "plot/"
    folders = ['smallP/']
    pattern = "*.MOV"

    for folder in folders:
        subdir = data_dir + folder
        if os.path.exists(subdir):
            for f in files(subdir):
                if fnmatch.fnmatch(f, pattern):
                    in_dir = subdir+f
                    file_name = folder + 'out' + f
                    plot_name = output_dir + f[0:len(f)-4] + ".png"
                    fc = FingerCounter("skin",\
                            in_dir, plot_name)
                    fc.play_game()
                    #fc_bgs = FC.FingerCounter("background",\
                    #        in_dir, bgs_out_dir)
                    #fc_bgs.play_game()
                
