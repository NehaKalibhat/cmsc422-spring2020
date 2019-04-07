from utils import *
from softmax import *

# Code adapted from https://github.com/jatinshah/ufldl_tutorial

# MNIST images are 28 * 28
exSize = 28*28
# 10 digits
numClasses = 10
# Regularizer coefficient
reg = 0.0001


X, Y = loadMNIST('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
sm = SoftmaxRegression(numClasses, exSize)
sm.train(X, Y)

testX, testY = loadMNIST('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
predictions = sm.predict(X)
print("Accuracy: {0:.2f}%".format(100 * np.sum(predictions == Y, dtype=np.float64) / Y.shape[0]))
