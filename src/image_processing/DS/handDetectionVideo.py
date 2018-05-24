import cv2
import numpy as np
import time


# TODO: 
# 1. Code refactoring
# 2. Error when no contours are found
# 3. 

def nothing(x):
    pass

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


def max_contour(thres):
 
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

    #Draw Contours
    # TODO: wrong contour drawed
    # frame = cv2.drawContours(frame, contour, -1, (122,122,0), 3)
    #cv2.imshow('contour',frame)

    #Largest area contour
    #print(contours)
    if (len(contours) != 0):
        cnts = contours[ci]
        #print(cnts)
        #time.sleep(10)
        return cnts
    else:
        return None


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
    cnts = max_contour(thres) 
    if (cnts is None):
        continue
    #Find contours of the filtered frame
    #Find convex hull
    hull = cv2.convexHull(cnts)

    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)

    #Get defect points and draw them in the original image
    FarDefect = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        FarDefect.append(far)
        cv2.line(frame,start,end,[0,255,0],1)
        cv2.circle(frame,far,10,[100,255,255],3)

	#Find moments of the largest contour
    moments = cv2.moments(cnts)

    #Central mass of first order moments
    if moments['m00']!=0:
        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    centerMass=(cx,cy)

    #Draw center mass
    cv2.circle(frame,centerMass,7,[100,0,255],2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)

    #Distance from each finger defect(finger webbing) to the center mass
    distanceBetweenDefectsToCenter = []
    for i in range(0,len(FarDefect)):
        x =  np.array(FarDefect[i])
        centerMass = np.array(centerMass)
        distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
        distanceBetweenDefectsToCenter.append(distance)

    #Get an average of three shortest distances from finger webbing to center mass
    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])

    #Get fingertip points from contour hull
    #If points are in proximity of 80 pixels, consider as a single point in the group
    finger = []
    for i in range(0,len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])

    #The fingertip points are 5 hull points with largest y coordinates
    finger =  sorted(finger,key=lambda x: x[1])
    fingers = finger[0:5]

    #Calculate distance of each finger tip to the center mass
    fingerDistance = []
    for i in range(0,len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
        fingerDistance.append(distance)

    #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    #than the distance of average finger webbing to center mass by 130 pixels
    result = 0
    for i in range(0,len(fingers)):
        if fingerDistance[i] > AverageDefectDistance+130:
            result = result +1

    #Print number of pointed fingers
    cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.drawContours(frame,[hull],-1,(255,255,255),2)

    ##### Show final image ########
    cv2.imshow('Dilation',frame)
    ###############################

    #Print execution time
    #print time.time()-start_time

    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
