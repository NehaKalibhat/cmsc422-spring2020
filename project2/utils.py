import numpy as np
from matplotlib import pyplot as plt


def loadMNIST(image_file, label_file):
    """
    returns a 28x28x[number of MNIST images] matrix containing
    the raw MNIST images
    :param filename: input data file
    """

    with open(image_file, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]
        num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)[0]

        exSize = num_rows * num_cols
        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((num_images, exSize)).transpose()
        images = images.astype(np.float64) / 255

        f.close()

    with open(label_file, 'r') as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        labels = np.fromfile(f, dtype=np.ubyte)

        f.close()

    return images, labels

def onehot(labels):
    """ From labels to one hot encoding"""
    y = np.zeros((len(labels), len(np.unique(labels[:]))), dtype=np.int8)
    y[np.arange(len(labels)), labels] = 1
    return y.transpose()

def data_loader(X, Y=None, batch_size=64, shuffle=False):
    """Iterator that yields one batch of data at a time"""
    N = X.shape[1]
    if shuffle:
        perm = np.random.permutation(N)
        X = X[:,perm]
        if Y is not None:
            Y = Y[:,perm]
    start = 0
    while start < N:
        end = start + batch_size
        if end > N:
            end = N
        if Y is not None:
            yield X[:, start:end], Y[:, start:end]
        else:
            yield X[:,start:end]
        start = end
    return

def acc(pred_label, Y):
    ''' pred_label: (N,) vector; Y: (N,K) one hot encoded ground truth'''
    num = len(pred_label)
    return sum(Y[pred_label[i], i] == 1 for i in range(num))*1.0/num

def save_plot(X, Y, pdf_name='learningcurve'):
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(X, Y)
    plt.savefig(pdf_name+'.pdf')

def raiseNotDefined():
  print("Method not implemented: %s" % inspect.stack()[1][3])
  sys.exit(1)

