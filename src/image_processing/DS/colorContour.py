import cv2
import numpy as np  
import time

eps = 1e-10

# reads video from video_dir and outputs processed video to output_dir
# returns rcs
def detect_hand(video_dir, output_dir):
    
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
        frame = cv2.flip(frame, 1)
        
        #detect hand with skin color
        skin_mask = detect_skin(frame)
        #thresh = morphological_transform(skin_mask)
        
        # write frame
        out.write(skin_mask)
        

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    rcs = 0
    return rcs

def find_contours(thresh, frame):
    
    _, contours, hierarchy = cv2.findContours(thresh,\
            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print('contour', contours)
    print('hierarch', heirarchy)
    #Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i

    #Draw Contours
    frame = cv2.drawContours(frame, cnt, -1, (122,122,0), 3)

    return frame


def morphological_transform(frame):
    
    # Kernel matrices for morphological transformation
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(frame, kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)

    return thresh, median


# outputs frame of skin masked 
def detect_skin(frame):
    
    # blur image
    blur = cv2.blur(frame, (5, 5))
    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2,50,50]), \
            np.array([15,255,255]))
    skin = cv2.bitwise_and(frame, frame, mask=mask)

    return skin




# cos angle between 2 vectors
# returns in degree format
def angle(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.sqrt((v1 * v1).sum() + eps)
    norm2 = np.sqrt((v2 * v2).sum() + eps)
    cos_angle = dot / (norm1 * norm2)
    deg = np.degrees(np.arccos(cos_angle))
    return deg


   
    
        





