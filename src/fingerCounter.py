import cv2
import skin_detection as sd

class FingerCounter():

    def __init__(self, mode, \
            in_dir=None, out_dir=None):
        
        if mode == 'background':
            self.is_background = True
        elif mode == 'skin':
            self.is_background = False
        else:
            raise UnavialableModeError()

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.is_webcam = True
        if in_dir is not None:
            self.is_webcam = False
        
    def play_game():
        cap = None
        bgs = BackgroundSubtractor(self.is_webcam)

        if not self.is_webcam:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.in_dir)
            fourcc = cv2.VideoWriter_fourcc(*'XVID') 
            out = cv2.VideoWriter(self.out_dir)
       
        while cap.open():
        
            ret, frame = cap.read()
            if self.is_background:
                mask = bgs(frame)
            else:
                mask = sd.detect_skin()

            #finger_cnt, frame = get_finger(mask)

            if self.is_webcam:
                cv2.imshow()
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

class UnavailableModeError(Exception):
    
    def __str__(self):
        return "only 'skin' or 'background' is available"
