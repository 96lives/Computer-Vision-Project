import cv2
import numpy as np
import skin_detection as sd

def frame2data(frame):
    
    mask = sd.mask_skin(frame)
    skin_area = cv2.bitwise_and(frame, frame, mask=mask)
    coordList = np.argwhere(mask != 0)
    skin_area = np.reshape(skin_area,(-1, 3))
    frame = np.reshape(frame, (-1, 3))
    zeros = np.zeros((1, 3), dtype=int)
    positive = skin_area[sum(skin_area != zeros)]
    pos_idx = [any(skin_area[i]!=0) for i in range(skin_area.shape[0])]

    neg_idx = [all(skin_area[i]==0) for i in range(skin_area.shape[0])]
    positive = np.unique(skin_area[pos_idx], axis=0)
    negative = np.unique(frame[neg_idx], axis=0)
    '''
    for i in coordList:
        x = i[0]
        y = i[1]
        positive.append(frame[x, y])
    coordList = np.argwhere(mask == 0)
    negative = []
    for i in coordList:
        x = i[0]
        y = i[1]
        negative.append(frame[x, y])
        '''
    return positive, negative

img_dir = "hand_photo.jpg"
frame = cv2.imread(img_dir, 1)
#print(frame.shape)

mask = frame2data(frame)

#cv2.imshow('mask', mask)
#cv2.waitKey()
'''
    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
'''


'''
video_dir = "../data/smallP/IMG_0037.MOV"

cap = cv2.VideoCapture(video_dir)

cnt = 0
while(True):
    
    ret, frame = cap.read()
    if (cnt == 0):
        frame = cv2.flip(frame,1)
        frame = cv2.flip(frame,0)
        cv2.imwrite('hand_photo.jpg', frame)
        break
cap.release()
cv2.destroyAllWindows()
'''


