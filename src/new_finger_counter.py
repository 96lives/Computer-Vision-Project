import cv2
import math
import skin_detection as sd
import diff_shaker as sh
import matplotlib.pyplot as plt
import time
from skin_color_classifier import SkinColorClassifier
from visualize import visualizer

frame_size = (240, 180)
#frame_size = (320, 240)

class NewFingerCounter():

    def __init__(self, video_name, \
            report_name, in_dir, \
            out_dir, save_video=False):
        
        self.video_name = video_name
        self.report_name = report_name
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.save_video = save_video

    def play_game(self):

        shaker = sh.DiffShaker()
        scc = None
        shake_switch = False
        shake_ended = False
        cnt_list = []
        vis = visualizer()

        cap = cv2.VideoCapture(self.in_dir+self.video_name)
        print(self.in_dir + self.video_name)
        frame_cnt = 0
        f = open(self.out_dir + self.report_name, 'a')
        f.write(self.video_name + ": ")
        pure_video_name = self.video_name.replace('.MOV', '')
        decision_cnt = 0
        finger_cnt = 0

        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(
                    self.out_dir + pure_video_name + "out.avi",\
                    fourcc, round(cap.get(5)), \
                    frame_size)
        
        
        ret, prev_frame = cap.read()
        prev_frame = render_frame(prev_frame)

        while cap.isOpened():

            ret, curr_frame = cap.read()
            frame_cnt += 1

            if ret is False:
                break
            
            curr_frame = render_frame(curr_frame)

            if shake_ended is True:
                if shake_switch is False:
                    # only started once
                    print('shake ended')
                    #time.sleep(2)
                    shake_switch = True
                    img1, img2 = shaker.get_minmax_image()
                    cv2.imwrite(self.out_dir + pure_video_name + '_max.jpg', img1)
                    cv2.imwrite(self.out_dir + pure_video_name + '_min.jpg', img2)
                    f.write(str(frame_cnt))
                    scc = SkinColorClassifier(img1, img2)
                start_time = time.time()
                mask = scc.mask_image(curr_frame)
                curr_frame, finger_cnt = count_finger(curr_frame, mask)
                print(time.time() - start_time)
            else:
                mask = sd.detect_skin(curr_frame)
                decision_cnt += 1

            if shake_switch is False:
                mask, shake_ended = \
                        shaker.shake_detect(prev_frame, curr_frame)
            cv2.imshow('mask', mask)
            if self.save_video:
                out.write(cv2.cvtColor(mask,\
                        cv2.COLOR_GRAY2BGR))


            prev_frame = curr_frame
            
            curr_frame = vis.visualize(curr_frame, finger_cnt, decision_cnt)
            cv2.imshow('frame', curr_frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        f.write('\n')
        f.close()
        if self.save_video:
            out.release()
        plt.plot(shaker.yhistory)
        plt.ylabel('avg y')
        
        plt.plot(shaker.smoothed)
        plt.ylabel('smoothed')
        plt.savefig(self.out_dir + pure_video_name + "_plot.png")
        plt.clf()

        plt.plot(cnt_list)
        plt.savefig(self.out_dir + pure_video_name + \
                "_finger_plot.png")
        plt.clf()

        cap.release()
        cv2.destroyAllWindows()
        return cnt_list


def render_frame(frame):
    frame = cv2.resize(frame, frame_size)
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)
    return frame
    

class UnavailableModeError(Exception):
    
    def __str__(self):
        return "only 'skin' or 'background' is available"


def count_finger(frame, mask):
    if mask is None:
        return frame, 0
    max_contour = find_max_contour(mask)
    if max_contour is None: 
        return frame, 0

    x,y,w,h = cv2.boundingRect(max_contour)
    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    hull = cv2.convexHull(max_contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(max_contour, hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])
                a = compute_distance(end, start)
                b = compute_distance(far, start)
                c = compute_distance(end, far)
                angle = compute_angle(a, b, c)
                # treat fingers with angle <= 90
                if angle <= (math.pi / 2):  
                    cnt += 1
                    frame = cv2.circle(frame, \
                            far, 8, [211, 84, 0], -1)
            return frame, cnt
    return frame, 0

def find_max_contour(mask):
 
    _, contours, hierarchy = cv2.findContours(mask,\
            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Find Max contour area (Assume that hand is in the frame)
    max_area=400
    ci=0
    contour = -1
    if contours is None:
        return None
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if(area>max_area):
            max_area=area
            ci=i

    #Largest area contour
    if (len(contours) != 0):
        max_contour = contours[ci]
        return max_contour
    else:
        return None

# computes the L2 distance of tuple a and b
def compute_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 \
            + (a[1] - b[1]) ** 2)

def compute_angle(a, b, c):
    eps = 1e-10
    return math.acos((b**2 + c**2 - a**2) \
        / (3*b*c + eps))
