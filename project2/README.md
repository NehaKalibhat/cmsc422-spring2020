# Project 2: PCA, Softmax Regression, and Neural Networks

In this project, we will explore dimensionality reduction (PCA), softmax regression and neural networks.

Files to turn in:

```
PCA.ipynb       Implementation of PCA
softmax.py      Implementation of softmax regression
nn.py           Implementation of fully connected neural network
writeup.pdf     Answers all the written questions in this assignment
```

You will be using helper functions in the following `.py` files and datasets:

```
datasets.py     Helper functions for PCA
utils.py        Helper functions for Softmax Regression and NN
digits          Toy dataset for PCA
data/*          Training and dev data for SR and NN
```

## Part 1 - Principal Component Analysis (PCA) [35%]

### 1.1 - Implement PCA [15%]

Our first tasks are to implement PCA. If implemented correctly, these should be 5-line functions (plus the supporting code I've provided): just be sure to use `numpy`'s eigenvalue computation code. Implement PCA in the function `pca` in [`PCA.ipynb`](PCA.ipynb).

The pseudo-code in [Algorithm 37 in CIML](http://ciml.info/dl/v0_99/ciml-v0_99-ch15.pdf) demonstrates the role of covariance matrix in PCA. However, the implementation of covariance matrix in practice requires much more concerns. One of them is to decide whether we require an unbiased estimation of the covariance matrix, i.e. normalize `D` by `N-1` instead of `N` (biased). Even the popular packages, such as matlab and sklearn, differ in the implementation. To make things easy, we'll require the submitted code to implement an unbiased version.

### 1.2 - Visualization of MNIST [5%]

Implement the function `draw_digits` in [`PCA.ipynb`](PCA.ipynb). Here,
[`matplotlib`](https://matplotlib.org/) will be useful for you.

### 1.3 - Normalized Eigenvalues [10%]

Plot the normalized eigenvalues for the MNIST digits. How many eigenvectors do you have to include before you've accounted for 90% of the variance? 95%?

### 1.4 - Visualization of Dimensionality Reduction [5%]

Plot the top 50 eigenvectors. Do these look like digits? Should they? Why or why not?

### Part 1 Hints

1. Read reference 2.
2. [`np.linalg.eig`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html), [`np.argsort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html), and [`np.cumsum`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.cumsum.html) will be of use.
3. Take the real components of the eigenvalues and eigenvectors.

### Restrictions

The use of `sklearn.decomposition.PCA` or `numpy.cov` is prohibited.

### Part 1 References

1. [PCA Tutorial](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)
2. [Mathematics of PCA](https://www.stat.cmu.edu/~cshalizi/uADA/16/lectures/17.pdf)
3. [Sample Mean and Covariance](https://en.wikipedia.org/wiki/Sample_mean_and_covariance)
4. [Eigenpictures](http://engr.case.edu/merat_francis/EECS%20490%20F04/References/Face%20Recognition/LD%20Face%20analysis.pdf)

## Part 2 - Softmax Regression [45%]

For this part of the project, you'll be working with [`softmax.ipynb`](softmax.ipynb).
Write your answers in the notebook.

### 2.1 - Questions about the Softmax Function [5%]

For both problems, assume there are `C` classes, `n` be the number of samples, and `d` be the number of features for each sample.

1. Prove that the probabilities outputed by the softmax function sum to 1.
2. Given the description of matrices `W`, `X` above, what are the dimensions of `W`, `X`, and `WX`? (Note that the description is provided in the notebook.)

### 2.2 - Implementing a Softmax Classifier [15%]

Implement `cost` and `predict` functions in the `SoftmaxRegression` class provided.
You can check the correctness of your implementation in the notebook.

### 2.3 - Stability [10%]

In the `cost` function of `SoftmaxRegression`, we see the line

```python3
W_X = W_X - np.amax(W_X)
```

1. What is this operation doing?
2. Show that this does not affect the predicted probabilities.
3. Why might this be an optimization? Justify your answer.

### 2.4 - Analysis of Classifier Accuracy [10%]

Plot the accuracy of the classifier as a function of the number of examples seen.
Do you observe any overfitting or underfitting? Discuss and expain what you observe.

### Part 2 References

1. [Softmax and its Derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

### Part 2 Hints

1. What happens when you take the exponential of a large positive number? A large negative number?
2. Again, use [`matplotlib`](https://matplotlib.org/) for 2.4.

Part III NN [25% and Extra 15%]
Files to edit/turn in.

nn.py
writeup.pdf
files created for the Q2 extra credits
In this part of the project, we'll implement a fully-connected neural network in general for the MNIST dataset. For Qnn1  you will complete nn.py. For Qnn2, create your own files.

A code stub also has been provided in run_nn.py. Once you correctly implement the incomplete portions of nn.py, you will be able to run run_nn.py in order to classify the MNIST digits.

 The dataset is included under ./data/. You will be using the following helper functions in "utils.py". In the following description, let LaTeX: K,\:N,\:d K , N , d  denote the number of classes, number of samples and number of features.

Selected functions in "utils.py"

loadMNIST(image_file, label_file) #returns data matrix X with shape (d, N) and labels with shape (N,)
onehot(labels) # encodes labels into one hot style with shape (K,N)
acc(pred_label, Y) # calculate the accuracy of prediction given ground truth Y, where pred_label is with shape (N,), Y is with shape (N,K).
data_loader(X, Y=None, batch_size=64, shuffle=False) # Iterator that yield X,Y with shape(d, batch_size) and (K, batch_size).
Qnn1 (20% for Qnn1.1, 1.2, 1.3 and 5% for Qnn 1.4) Implement the NN

The scaffold has been built for you. Initialize the model and print the architecture with the following:

 >>> from nn import NN, Relu, Linear, SquaredLoss
 >>> from utils import data_loader, acc, save_plot, loadMNIST, onehot
 >>> model = NN(Relu(), SquaredLoss(), hidden_layers=[128,128])
 >>> model.print_model()
Two activation functions (Relu, Linear) and self.predict(X) have been implemented for you.

Qnn1.1 Implement squared loss cost functions (TODO 0 & TODO 1)
Assume LaTeX: \bar Y Y ¯  is the output of the last layer before loss calculation (without activation), which is a K-by-N matrix. LaTeX: Y Y  is the one hot encoded ground truth of the same shape. Implement the following loss function and its gradient (You need to calculate and implement the gradient of the loss function yourself) (Notice that the loss functions are normalized by batch_size LaTeX: N N ):
LaTeX: L\left(Y,\bar Y\right) = \frac{1}{N}\sum_{i=1}^N \frac{1}{2}\|\bar Y_i - Y_i\|^2 = \frac{1}{2N}\|\bar Y - Y\|^2_{fro} L ( Y , Y ¯ ) = 1 N ∑ i = 1 N 1 2 ‖ Y ¯ i − Y i ‖ 2 = 1 2 N ‖ Y ¯ − Y ‖ f r o 2 , where LaTeX: Y_i Y i  is the LaTeX: i i -th column of LaTeX: Y Y .

Typically we would use cross entropy loss, the formula of which is provided for your reference (but you are only required to implement for squared loss):

LaTeX: L_{CE}(Y,\bar Y) = \frac{1}{N}\sum_{i=1}^N NLL(Y_i, \bar Y_i) L C E ( Y , Y ¯ ) = 1 N ∑ i = 1 N N L L ( Y i , Y ¯ i ) , where

LaTeX: NLL(y, \bar y) = -\log (\sum_{j=1}^K y_j \frac{\exp(\bar y_j)}{\sum_{k=1}^K \exp(\bar y_k)}) N L L ( y , y ¯ ) = − log ⁡ ( ∑ j = 1 K y j exp ⁡ ( y ¯ j ) ∑ k = 1 K exp ⁡ ( y ¯ k ) ) .

Qnn1.2. Compute the gradients (TODO 2 & TODO 3)
Implement the forward pass (TODO 2) and back propagation (TODO 3) for gradient calculation. Use "activation.activate" and "activation.backprop_grad" in your code so that your gradient computation works for different choices of activation functions.
Do the following to see if the loss goes down.

>>> x_train, label_train = loadMNIST('data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte')
>>> x_test, label_test = loadMNIST('data/t10k-images.idx3-ubyte', 'data/t10k-labels.idx1-ubyte')
>>> y_train = onehot(label_train)
>>> y_test = onehot(label_test)

>>> model = NN(Relu(), SquaredLoss(), hidden_layers=[128, 128], input_d=784, output_d=10)
>>> model.print_model()
>>> training_data, dev_data = {"X":x_train, "Y":y_train}, {"X":x_test, "Y":y_test}
>>> from run_nn import train_1pass
>>> model, plot_dict = train_1pass(model, training_data, dev_data, learning_rate=1e-2, batch_size=64)
Qnn1.3 Run in epochs

An epoch is a full pass of the training data. Run run_nn.py.

>>> python3 run_nn.py
Report your final accuracy on the dev set. You can either use the default setting or tune the architecture (number of layers, size of layers and loss function) and hyperparameters (lr, batch_size, max_epoch).

Qnn1.4 (No implementation needed for this question). When initializing the weight matrix, in some cases it may be appropriate to initialize the entries as small random numbers rather than all zeros.  Give one reason why this may be a good idea.

Qnn2 (Extra-Credit 15%) Try something new.

Choose one of the following directions (outside research may be required) for further exploration (Feel free to copy nn.py, utils.py and run_nn.py as a starting point. Make sure that your code for Qnn2 is separated from your code for Qnn1):


(1) Do dimension reduction with PCA. Try with different dimensions. Can you observe the trade-off in time and acc? Plot training time v.s. dimension, testing time v.s dimension and acc v.s. dimension. Visualize the principal components.

(2) Improve your results with ensemble methods. Describe your implementation. Can you observe improved performance compared with that of Q4? Why? (http://ciml.info/dl/v0_99/ciml-v0_99-ch13.pdf)

(3) Implement a new optimizer (By implementing a different self.update for the NN class). Compare with the original SGD optimizer. You can read about the optimizers in (http://ruder.io/deep-learning-optimization-2017/). Does this new method take less number of samples to converge? Does this new method take less time to converge?

Qnn2.1 Explain what you did and what you found. Comment the code so that it is easy to follow. Support your results with plots and numbers. Provide the implementation so we can replicate your results.