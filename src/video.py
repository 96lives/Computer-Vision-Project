import os 
import fnmatch
import skin_detection as sd
import BackgroundSubtractor as BGS
import fingerCounter as FC
import cv2

file_name = "test.MOV"
out_dir = "output.avi"
cap = cv2.VideoCapture(file_name)
fc = FC.FingerCounter('skin', file_name, out_dir)
fc.play_game()

'''
file_name = "test.MOV"
out_dir = "output.avi"
cap = cv2.VideoCapture(file_name)
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
print(cap.get(5))
out = cv2.VideoWriter(out_dir, fourcc, round(cap.get(5)),
        (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    #cv2.imshow('orignal frame', frame)
    if ret:
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        #print(frame.shape)
        mask = sd.detect_skin(frame)
        frame, finger_cnt = FC.count_finger(frame, mask)

        out.write(frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
'''

'''
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

if __name__ == "__main__":
    data_dir = "../data/"
    output_dir = "../data/output/"
    folders = ['R/' , 'P/', 'S/']
    pattern = "*.MOV"

    for folder in folders:
        subdir = data_dir + folder
        if os.path.exists(subdir):
            for f in files(subdir):
                if fnmatch.fnmatch(f, pattern):
                    in_dir = subdir+f
                    file_name = folder + 'out' + f
                    skin_out_dir = output_dir\
                            + "skin/" + file_name
                    #bgs_out_dir = output_dir\
                    #        +"bgs/" + file_name
                    fc_skin = FC.FingerCounter("skin",\
                            in_dir, skin_out_dir)
                    fc_skin.play_game()
                    #fc_bgs = FC.FingerCounter("background",\
                    #        in_dir, bgs_out_dir)
                    #fc_bgs.play_game()
'''
                