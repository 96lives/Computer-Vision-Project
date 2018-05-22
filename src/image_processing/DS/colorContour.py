import cv2
import numpy as np  
import time

eps = 1e-10

# reads video from video_dir and outputs processed video to output_dir
# returns rcs
def detect_skin_color(video_dir, output_dir):
    
    # open camera object
    cap = cv2.VideoCapture(video_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_dir, fourcc, 30 \
            ,(int(cap.get(3)), int(cap.get(4))) )
    
    while (cap.isOpened()):
        
        # ret is whether frame is true or not
        ret, frame = cap.read()
        if (not ret):
            break
        
        # flip the image
        frame = cv2.flip(frame, 0)
        # do some image processing
        frame = detect_hand(frame)
        # detect hand
        out.write(frame)
        

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    rcs = 0
    return rcs

# outputs frame of image processed hand 
def detect_hand(frame):
    
    # blur image
    blur = cv2.blur(frame, (5, 5))
    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2,50,50]), \
            np.array([15,255,255]))
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return frame


# cos angle between 2 vectors
# returns in degree format
def angle(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.sqrt((v1 * v1).sum() + eps)
    norm2 = np.sqrt((v2 * v2).sum() + eps)
    cos_angle = dot / (norm1 * norm2)
    deg = np.degrees(np.arccos(cos_angle))
    return deg


   
    
        





