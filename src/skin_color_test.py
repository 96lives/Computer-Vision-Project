import cv2
import numpy as np
import skin_detection as sd
import SkinColorClassifier as SCC

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
frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

#print(frame.shape)

def classify_pixel(pixel, scc):
    if scc.classify(pixel) == 1:
        return [255, 0, 0]
    return pixel

w, h, channel = frame.shape
pos, neg = frame2data(frame)
scc = SCC.SkinColorClassifier(pos, neg)
frame = frame.reshape(-1, 3)
mask = scc.classify(frame)
mask = np.asarray(mask).reshape(w, h)
print(mask.shape)
mask[mask>0] = 255
mask[mask<0] = 0


#frame = frame.swapaxes(2, 0).copy(order='C')
#print(frame)
#for x in np.nditer(frame, flags=['external_loop'], op_flags=['readwrite'], order='F'):
    #classify_pixel(x, scc)
    #print(x)
#    print(classify_pixel(x, scc))

'''
for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
        print(str(i) + ',' + str(j))
        if scc.classify(frame[i, j]) == 1:
            new_image[i, j] = [255, 0, 0]
        else:
            new_image[i, j] = [0, 0, 0]
'''
cv2.imshow('mask', mask)
cv2.waitKey()
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


