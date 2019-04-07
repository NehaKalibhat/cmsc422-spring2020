import numpy as np
from nn import NN
from nn import Relu, Linear, SquaredLoss, CELoss
from utils import data_loader, acc, save_plot, loadMNIST, onehot

# Several passes of the training data
def train(model, training_data, dev_data, learning_rate, batch_size, max_epoch):
    X_train, Y_train = training_data['X'], training_data['Y']
    X_dev, Y_dev = dev_data['X'], dev_data['Y']
    for i in range(max_epoch):
        for X,Y in data_loader(X_train, Y_train, batch_size=batch_size, shuffle=True):
            training_loss, grad_Ws, grad_bs = model.compute_gradients(X, Y)
            model.update(grad_Ws, grad_bs, learning_rate)
        dev_acc = acc(model.predict(X_dev), Y_dev)
        print("Epoch {: >3d}/{}\tloss:{:.5f}\tdev_acc:{:.5f}".format(i+1,max_epoch,training_loss, dev_acc))
    return model

# One pass of the training data
def train_1pass(model, training_data, dev_data, learning_rate, batch_size, print_every=100, plot_every=10):
    X_train, Y_train = training_data['X'], training_data['Y']
    X_dev, Y_dev = dev_data['X'], dev_data['Y']

    num_samples = 0
    print_loss_total = 0
    plot_loss_total = 0
    
    plot_losses = []
    plot_num_samples = []
    for idx, (X,Y) in enumerate(data_loader(X_train, Y_train, batch_size=batch_size, shuffle=True),1):
        training_loss, grad_Ws, grad_bs = model.compute_gradients(X, Y)
        model.update(grad_Ws, grad_bs, learning_rate)
        num_samples += Y.shape[1]
        print_loss_total += training_loss
        plot_loss_total += training_loss
        
        if idx % print_every == 0:
            dev_acc = acc(model.predict(X_dev), Y_dev)
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print("#Samples {: >5d}\tloss:{:.5f}\tdev_acc:{:.5f}".format(num_samples, print_loss_avg, dev_acc))
        if idx % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)
            plot_num_samples.append(num_samples)
        
    return model, {"losses":plot_losses, "num_samples":plot_num_samples}

if __name__ == "__main__":
    x_train, label_train = loadMNIST('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
    x_test, label_test = loadMNIST('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
    y_train = onehot(label_train)
    y_test = onehot(label_test)

    model = NN(Relu(), SquaredLoss(), hidden_layers=[256, 256], input_d=784, output_d=10)
    model.print_model()

    lr = 1e-2
    max_epoch = 20
    batch_size = 128
    training_data = {"X":x_train, "Y":y_train}
    dev_data = {"X":x_test, "Y":y_test}

    #model, plot_dict = train_1pass(model, training_data, dev_data, lr, batch_size)
    #save_plot(plot_dict["num_samples"], plot_dict["losses"]) 

    model = train(model, training_data, dev_data, lr, batch_size, max_epoch)
