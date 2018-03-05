import os
import struct
import numpy as np
import time

def read(dataset = "training", path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)
        
        
test_size = 8000
train_size = 40000


training = list(read(dataset='training'))
test = list(read(dataset='testing'))
test = test[:test_size]
training = training[:train_size]
train_data = []
train_label = []
test_data = []
test_label = []

def reduce_dimens(array):
    res = []
    for i in range(len(array)):
        for j in range(len(array[0])):
            res.append(array[i][j])
    return res

stime = time.time()
def now():
    return time.time() - stime
# print "Test data:\n\t", now()
for i in range(len(test)):
    test_data.append(reduce_dimens(test[i][1]))
    test_label.append(test[i][0])
    
    
# print "Train data:\n\t", now()
for i in range(len(training)):
    # new_data = []
    # for j in range(len(training[i][1])):
    #     for k in range(len(training[i][1][j])):
    #         new_data.append(training[i][1][j][k])
    # if (i == 1):
    #     print new_data
    # train_data.append(new_data)
    train_data.append(reduce_dimens(training[i][1]))
    train_label.append(training[i][0])
    
from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(10)

# print "training...:\n\t", now()
clf.fit(train_data, train_label)
# print "training done!",now()
# if (clf.predict([train_data[0]]) == training[1100][0]):
counter = 0
# print "predicting...:\n\t", now()
pred = clf.predict(test_data)

# print "testing data...:\n\t", now()
for i in range(len(test)):
    if (pred[i] == test_label[i]):
        counter += 1

print "accuracy is: ", float(counter) / float(len(test)), now()