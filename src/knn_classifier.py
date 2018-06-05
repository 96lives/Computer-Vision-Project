from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

class KNNClassifier():

    def __init__(self, positive, negative, num_neighbors=15): # n by 3(hsv) matrix
        positive = np.array(positive)
        negative = np.array(negative)
        y_positive = np.ones((positive.shape[0],1))
        y_negative = -np.ones((negative.shape[0],1))
        positive = np.concatenate((positive, y_positive), axis=1)
        negative = np.concatenate((negative, y_negative), axis=1)
        data = np.concatenate((positive, negative), axis=0)
        np.random.shuffle(data)
        self.X_train = data[:,:-1]
        self.y_train = data[:,-1]
        self.neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
        tic = time.time()
        self.neigh.fit(self.X_train, self.y_train)
        print(time.time()-tic)

    def classify(self, points):
        tic = time.time()
        points = np.array(points)
        channel = points.shape[1]
        points = points.reshape(-1,channel)
        print(time.time()-tic)
        return self.neigh.predict(points)

    def train(self, positive, negative):
        positive = np.array(positive)
        negative = np.array(negative)
        y_positive = np.ones((positive.shape[0],1))
        y_negative = -np.ones((negative.shape[0],1))
        positive = np.concatenate((positive, y_positive), axis=1)
        negative = np.concatenate((negative, y_negative), axis=1)
        data = np.concatenate((positive, negative), axis=0)
        np.random.shuffle(data)
        self.X_train = np.concatenate( (self.X_train, data[:,:-1]), axis=0)
        self.y_train = np.concatenate( (self.y_train, data[:,-1]), axis=0)
        self.neigh.fit(self.X_train, self.y_train)


if __name__ == '__main__':
    pdata = [[1, 2],[1, 3],[1, 4],[1, 5],[1, 6],[1, 7]]
    ndata = [[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7]]
    knc = KNNClassifier(pdata, ndata, num_neighbors=3)
    answer = knc.classify([[1.9,2],[2.0,5],[2.1,8]])
    print(answer)
    pdata = [[2, 2],[2, 3],[2, 4],[2, 5],[2, 6],[2, 7]]
    ndata = [[4, 2],[4, 3],[4, 4],[4, 5],[4, 6],[4, 7]]
    knc.train(pdata,ndata)
    answer = knc.classify([[1.9,2],[2.0,5],[2.1,8]])
    print(answer)

