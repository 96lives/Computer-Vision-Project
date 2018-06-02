import cv2
import numpy as np
import time
# outputs frame of skin masked 

def detect_skin(frame):
    frame = mask_skin(frame)
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


def morphological_transform(frame):
    
    # Kernel matrices for morphological transformation
    kernel_square = np.ones((21,21),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1, 1))

    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    #dilation = cv2.dilate(frame, kernel_ellipse,iterations = 1)
    erosion = cv2.erode(frame,kernel_square,iterations = 1)
    #dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    cv2.imshow('erosion', erosion)
    #filtered = cv2.medianBlur(dilation2,11)
    #kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    #dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(erosion,5)
    cv2.imshow('median', median)
    #ret,thres = cv2.threshold(median,127,255,0)

    # check is the thres is GBR format
    if len(median.shape) == 3:
        median = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY) 
    return median


def find_max_contour(thres):
 
    _, contours, hierarchy = cv2.findContours(thres,\
            cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    contour = -1
    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if(area>max_area):
            max_area=area
            ci=i

   #Largest area contour
    #print(contours)
    if (len(contours) != 0):
        max_contour = contours[ci]
        return max_contour
    else:
        return None

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
