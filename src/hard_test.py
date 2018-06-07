import cv2
import skin_detection as sd
import matplotlib.pyplot as plt
import numpy as np


def frame_diff(f1, f2):
    return abs(f1 - f2)


# resize and flip frames
def render_frame(frame):
    frame_size = (320, 240)
    frame = cv2.resize(frame, frame_size)
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)
    frame = cv2.medianBlur(frame, 21)
    return frame

def erode_frame(mask):
    kernel_ellipse7= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kernel_ellipse5= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    kernel_ellipse3= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    out = cv2.erode(mask,kernel_ellipse7,iterations = 1)    
    #dilation = cv2.dilate(erosion,kernel_ellipse5,iterations = 1)
    
    #kernel_ellipse3= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #erosion = cv2.dilate(mask,kernel_ellipse3,iterations = 1)    
    #dilation = cv2.erode(erosion,kernel_ellipse5,iterations = 1)
    out = cv2.medianBlur(out,7)
    return out
   


if __name__ == "__main__":

    video_dir = "../data/hardP/"
    video_name = "IMG_0162.MOV"
    out_dir = "../data/"

    cap = cv2.VideoCapture(video_dir + video_name)
    frame_size = (320, 240)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')             
    out = cv2.VideoWriter(\
            out_dir + video_name + "_out.avi",\
            fourcc, round(cap.get(5)), \
            frame_size)

    ret, prev_frame = cap.read()
    prev_frame = render_frame(prev_frame)
    frame_cnt = 0
    threshold = 0.8 * 255

    yhistory1 = []
    yhistory2 = []

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if ret:
            

            curr_frame = render_frame(curr_frame)
            mask = sd.mask_skin(curr_frame)
            diff = frame_diff(curr_frame, prev_frame)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff[diff < threshold] = 0
            diff = cv2.bitwise_and(diff, diff, mask=mask)
            diff = erode_frame(diff)
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            
            idx1 = np.argwhere(diff > 0)
            cnt1 = len(idx1)
            y_sum1 = 0
            for i in idx1:
                y_sum1 += i[0]
            if cnt1 is not 0:
                yhistory1.append(y_sum1/cnt1)
            elif len(yhistory1) != 0:
                yhistory1.append(yhistory1[-1])
            
            idx2 = np.argwhere(mask > 0)
            cnt2 = len(idx2)
            y_sum2 = 0
            for i in idx2:
                y_sum2 += i[0]
            if cnt2 is not 0:
                yhistory2.append(y_sum2/cnt2)

            cv2.imshow('diff', diff)
            cv2.imshow('mask', mask)
            k = cv2.waitKey(0)
            if k == 27:
                break
        
            out.write(diff)
            prev_frame = curr_frame
        elif not ret:
            break
        frame_cnt += 1

    plt.plot(yhistory1)
    plt.plot(yhistory2)
    plt.show()
    out.release()
    cap.release()
    cv2.destroyAllWindows()


 

















