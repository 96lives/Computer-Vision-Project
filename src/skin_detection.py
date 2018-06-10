import cv2
import numpy as np
import time
# outputs frame of skin masked 

def detect_skin(frame):
    frame = mask_skin(frame)
    cv2.imshow('skin mask', frame)
    return morphological_transform(frame)

def mask_skin(frame):
    
    # blur image
    #blur = cv2.blur(frame, (9, 9))
    lower = np.array([0, 48, 50], dtype="uint8")
    upper = np.array([25, 255, 255], dtype="uint8")
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    #skin = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('skin', skin)
    return mask


def morphological_transform(mask):

    kernel_ellipse7= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kernel_ellipse5= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erosion = cv2.erode(mask,kernel_ellipse7,iterations = 1)    
    dilation = cv2.dilate(erosion,kernel_ellipse5,iterations = 1)
    
    kernel_ellipse3= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.dilate(mask,kernel_ellipse3,iterations = 1)    
    dilation = cv2.erode(erosion,kernel_ellipse5,iterations = 1)
    median = cv2.medianBlur(dilation,5)
    
    if len(median.shape) == 3:
        median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY) 
    return median

def check_finger(frame, max_contour):
    #Find contours of the filtered frame
    #Find convex hull
    hull = cv2.convexHull(max_contour)

    #Find convex defects
    hull2 = cv2.convexHull(max_contour, returnPoints = False)
    defects = cv2.convexityDefects(max_contour, hull2)

    #Get defect points and draw them in the original image
    FarDefect = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        FarDefect.append(far)
        cv2.line(frame,start,end,[0,255,0],1)
        cv2.circle(frame,far,10,[100,255,255],3)
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    return frame

# Open Camera object
'''
data_dir = '../data/test.MOV'
cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    # Capture frames from the camera
    ret, frame = cap.read()
    if (not ret):
        break
    skin = detect_skin(frame)
    thres = morphological_transform(skin)
    cv2.imshow('skin', thres)
    max_contour = find_max_contour(thres) 
    if max_contour is not None:
        frame = check_finger(frame, max_contour)
        x,y,w,h = cv2.boundingRect(max_contour)
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    ##### Show final image ########
    cv2.imshow('Dilation',frame)
    ###############################

    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
'''
