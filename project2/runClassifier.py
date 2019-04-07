"""
This module is for training, testing an evaluating classifiers.
"""

from numpy import *
from pylab import *


def trainTest(classifier, numClasses, exSize, X, Y, Xtest, Ytest):
    """
    Train a classifier on data (X,Y) and evaluate on
    data (Xtest,Ytest).  Return a triple of:
      * Training data accuracy
      * Test data accuracy
      * Individual predictions on Xtest.
    """
    classifier.train(X, Y);                      # train it

    #print "Learned Classifier:"
    #print classifier

    Ypred = classifier.predict(X);               # predict the training data
    trAcc = mean(Y == Ypred);                    # check to see how often the predictions are right

    Ypred = classifier.predict(Xtest);           # predict the training data
    teAcc = mean(Ytest == Ypred);                # check to see how often the predictions are right

    print ("Training accuracy %g, test accuracy %g" % (trAcc, teAcc))

    return (trAcc, teAcc, Ypred)

def learningCurve(classifier, numClasses, exSize, X, Y, Xtest, Ytest):
    """
    Generate a learning curve by repeatedly halving the amount of
    training data until none is left.

    We return a triple containing:
      * The sizes of data sets we trained on
      * The training accuracies at each level
      * The test accuracies at each level
    """

    N = X.shape[1]             # how many total points?
    print(N)
    M = int(ceil(log2(N)))     # how many classifiers will we have to train?
    print(M)

    dataSizes = zeros(M)
    trainAcc  = zeros(M)
    testAcc   = zeros(M)
    
    for i in range(1, M+1):    # loop over "skip lengths"
        # select every 2^(M-i)th point
        print(i)
        ids = arange(0, N, 2**(M-i))
        Xtr = X[:, ids]
        Ytr = Y[ids]

        print (Xtr.shape)
        print (Ytr.shape)

        # report what we're doing
        print ("Training classifier on %d points..." % ids.size)

        # train the classifier
        (trAcc, teAcc, Ypred) = trainTest(classifier, numClasses, exSize, Xtr, Ytr, Xtest, Ytest)
        
        # store the results
        dataSizes[i-1] = ids.size
        trainAcc[i-1]  = trAcc
        testAcc[i-1]   = teAcc

    return (dataSizes, trainAcc, testAcc)

def hyperparamCurve(classifier, hpName, hpValues, numClasses, exSize, X, Y, Xtest, Ytest):
    M = len(hpValues)
    trainAcc = zeros(M)
    testAcc  = zeros(M)
    for m in range(M):
        # report what we're doing
        print ("Training classifier with %s=%g..." % (hpName, hpValues[m]))
        
        # train the classifier
        classifier.reset(numClasses, exSize)
        classifier.setOption(hpName, hpValues[m])
        (trAcc, teAcc, Ypred) = trainTest(classifier, numClasses, exSize, X, Y, Xtest, Ytest)

        # store the results
        trainAcc[m] = trAcc
        testAcc[m]  = teAcc

    return (hpValues, trainAcc, testAcc)
