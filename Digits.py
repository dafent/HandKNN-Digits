import struct as st
import numpy as np
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#from matplotlib import pyplot as plt
import itertools
from collections import Counter

def read_idx(filename):
    with open (filename, 'rb') as f:
        zero, data_types, dims = st.unpack('>HBB', f.read(4))
        shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype = np.uint8).reshape(shape)
        
raw_train = read_idx("train-images-idx3-ubyte")
train_data = np.reshape(raw_train, (60000, 28*28))
train_label = read_idx("train-labels-idx1-ubyte")

raw_test = read_idx("t10k-images-idx3-ubyte")
test_data = np.reshape(raw_test, (10000, 28*28))
test_label = read_idx("t10k-labels-idx1-ubyte")

X = train_data
Y = train_label
X_test = test_data[:20]
Y_true = test_label[:20]

class My_knn:
    
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        
    def fit(self, train_x, train_y):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        
    def predict(self, test_x):
        pred_y = []
        count = 0
        
        for elem in test_x:
            dist = []
            pred_y_counter = []
            dist_kmin = []
            
            for ind, x in enumerate(self.train_x):
                dist.append((My_knn.distance(elem, x), ind))
                
            dist_kmin = sorted(dist, key = lambda x: x[0])[:self.k]
            pred_y_count = Counter(self.train_y[[item[1] for item in dist_kmin]])
            pred_y.append(pred_y_count.most_common(1)[0][0])
            print("count: {}, pred_y: {}".format(count, pred_y_count.most_common(1)[0][0]))
            #if count%100 == 0:
                #print("count: {}, pred_y: {}".format(count, pred_y[count]))
            count += 1
                
        return pred_y
    
    @staticmethod
    def distance(im_1, im_2):
        p = 100.
        im_1 = np.array(im_1)
        im_2 = np.array(im_2)
        #dist = np.sqrt(np.sum((im_1 - im_2)**2))
        #dist = np.sqrt(np.dot(im_1, im_2) - 2 * np.dot(im_1, im_2) + np.dot(im_1, im_2))
        #dist = np.abs(im_1 - im_2)
        dist = np.linalg.norm(im_1 - im_2)
        #dist = (np.sum((im_1 - im_2)**p))**(1/p)
        #dist = np.sum(np.abs(im_1 - im_2))/(np.sum(np.abs(im_2)) + np.sum(np.abs(im_1)))
        #dist = np.sum(im_1*np.log(im_1/im_2))
        return dist
        
KNN = My_knn(5)

KNN.fit(X, Y)

Y_pred = KNN.predict(X_test)
print(classification_report(Y_true, Y_pred))
