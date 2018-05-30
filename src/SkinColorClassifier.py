from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time

class SkinColorClassifier():

    def __init__(self, positive, negative): # n by 3(hsv) matrix
        positive = np.array(positive)
        negative = np.array(negative)
        y_positive = np.ones((positive.shape[0],1))
        y_negative = -np.ones((negative.shape[0],1))
        positive = np.concatenate((positive, y_positive), axis=1)
        negative = np.concatenate((negative, y_negative), axis=1)
        data = np.concatenate((positive, negative), axis=0)
        np.random.shuffle(data)
        X_train = data[:,:-1]
        y_train = data[:,-1]
        self.forest = RandomForestClassifier(n_estimators = 5, random_state = 2, n_jobs = -1)
        tic = time.time()
        self.forest.fit(X_train, y_train)
        print(time.time()-tic)

    def classify(self, point):
        #tic = time.time()
        point = np.array(point)
        point = point.reshape(1,-1)
        #print(time.time()-tic)
        return self.forest.predict(point)

if __name__ == '__main__':
    pdata = [[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],[1, 7]]
    ndata = [[2, 2],[2, 3],[2, 4],[2, 5],[2, 6],[2, 7]]
    pdata = np.random.rand(640*240,3)
    ndata = np.random.rand(640*240,3)
    scc = SkinColorClassifier(pdata, ndata)
    answer = scc.classify([2,2,2])
    print(answer)
