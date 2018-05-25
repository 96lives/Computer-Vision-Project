import cv2
import numpy as np
import time

# Function to find angle between two vectors
def Angle(v1,v2):
    dot = np.dot(v1,v2)
    x_modulus = np.sqrt((v1*v1).sum())
    y_modulus = np.sqrt((v2*v2).sum())
    cos_angle = dot / x_modulus / y_modulus
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B):
    return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2))


# outputs frame of skin masked 
def detect_skin(frame):
    
    # blur image
    blur = cv2.blur(frame, (5, 5))
    
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,30,60]), \
            np.array([20,150,255]))
    skin = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('skin', skin)
    return skin


# TODO: Do more clear image transform
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
    median = cv2.medianBlur(dilation2,5)
    ret,thres = cv2.threshold(median,127,255,0)
    thres = cv2.cvtColor(thres, cv2.COLOR_BGR2GRAY) 
    #cv2.imshow('thres', thres)
    return thres, median


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
data_dir = '../data/test.MOV'
cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    # Capture frames from the camera
    ret, frame = cap.read()
    if (not ret):
        break
    skin = detect_skin(frame)
    thres, median = morphological_transform(skin)
    max_contour = find_max_contour(thres) 

   if (max_contour is not None):
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
ap.release()
cv2.destroyAllWindows()
