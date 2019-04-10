# Project 2: PCA, Softmax Regression, and Neural Networks

In this project, we will explore dimensionality reduction (PCA), softmax regression and neural networks. You are allowed to work in groups, with a max size of 3.

Files to turn in:

```
p2_pca_softmax_nn.ipynb       Implementation of project
writeup.pdf                   Notebook exported as PDF via Latex
```

See the [how to submit](#how-to-submit) section for more details.

You will be using the following datasets:

```
data/*                        Training data for PCA and Softmax Regression
```

## Getting Started

1. Install [Conda](https://docs.anaconda.com/anaconda/install/). We'll use it to ensure that our environments are the same.
2. In this directory run `conda env create -f environment.yml`.
3. Run the environment using `conda activate cmsc422-p2`.
4. To view the [Jupyter Notebook](https://jupyter.org) files, run `jupyter notebook`.

To exit the environment, use `conda deactivate`.

Here's a [tutorial on using Jupyter](https://www.youtube.com/watch?v=HW29067qVWk) if you're not already familiar with the tool!

The entire project will be done in [`p2_pca_softmax_nn.ipynb`](p2_pca_softmax_nn.ipynb). Please write your answers in the notebook.

## Part 1 - Principal Component Analysis (PCA) [35%]

### 1.1 - Implement PCA [15%]

Our first tasks are to implement PCA. If implemented correctly, these should be 5-line functions (plus the supporting code I've provided): just be sure to use `numpy`'s eigenvalue computation code. Implement PCA in the function `pca`.

The pseudo-code in [Algorithm 37 in CIML](http://ciml.info/dl/v0_99/ciml-v0_99-ch15.pdf) demonstrates the role of covariance matrix in PCA. However, the implementation of covariance matrix in practice requires much more concerns. One of them is to decide whether we require an unbiased estimation of the covariance matrix, i.e. normalize `D` by `N-1` instead of `N` (biased). Even the popular packages, such as matlab and sklearn, differ in the implementation. To make things easy, we'll require the submitted code to implement an unbiased version.

#### Restrictions

The use of `sklearn.decomposition.PCA` or `np.cov` is prohibited.

### 1.2 - Visualization of MNIST [5%]

Implement the function `draw_digits`.
[`matplotlib`](https://matplotlib.org/) will be useful for you.

It must implement the following specifications:

1. Visualize `K` random samples from `X`, with no repeats.
2. Digits must be visualized on 5 column subplots.
3. A gray colormap must be used to visualize the digits.
4. The label of the digit should be added as white text on the upper left corner of the subplot, with font size 16.
5. The axes of each subplot should be turned off.

### 1.3 - Plotting Explained Variance [10%]

Plot the explained variance of the principal components, with x-axis being the number of principal components, and the y-axis being the percent variance explained. How many eigenvectors do you have to include before you've accounted for 90% of the variance?
95%? Label these points on your plot.

### 1.4 - Visualization of Dimensionality Reduction [5%]

Plot the top 50 eigenvectors. Do these look like digits? Should they? Why or why not?

### Part 1 Hints

1. Read reference 2.
2. [`np.linalg.eig`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html), [`np.argsort`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html), and [`np.cumsum`](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.cumsum.html) will be of use.
3. Take the real components of the eigenvalues and eigenvectors.

### Part 1 References

1. [PCA Tutorial](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)
2. [Mathematics of PCA](https://www.stat.cmu.edu/~cshalizi/uADA/16/lectures/17.pdf)
3. [Sample Mean and Covariance](https://en.wikipedia.org/wiki/Sample_mean_and_covariance)
4. [Eigenpictures](http://engr.case.edu/merat_francis/EECS%20490%20F04/References/Face%20Recognition/LD%20Face%20analysis.pdf)

## Part 2 - Softmax Regression [45%]

### 2.1 - Questions about the Softmax Function [5%]

For both problems, assume there are `C` classes, `n` be the number of samples, and `d` be the number of features for each sample.

1. Prove that the probabilities outputed by the softmax function sum to 1.
2. Given the description of matrices `W`, `X` above, what are the dimensions of `W`, `X`, and `WX`? (Note that the description is provided in the notebook.)

### 2.2 - Implementing a Softmax Classifier [20%]

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

### Part 2 Hints

1. What happens when you take the exponential of a large positive number? A large negative number?
2. Again, use [`matplotlib`](https://matplotlib.org/) for 2.4.

### Part 2 References

1. [Softmax and its Derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

## How to Submit

Tentatively, submission will be done through [Gradescope](http://gradescope.com). Here, you will be able to specify the group members you worked with.
