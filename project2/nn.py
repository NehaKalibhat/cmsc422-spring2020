import numpy as np

class NN:
    def __init__(self, activation_function, loss_function, hidden_layers=[1024], input_d=784, output_d=10):
        self.weights = []
        self.biases = []
        self.activation_function = activation_function
        self.loss_function = loss_function

        # Initialization of weights and biases
        d1 = input_d
        hidden_layers.append(output_d)
        for d2 in hidden_layers:
            self.weights.append(np.random.randn(d2, d1)*np.sqrt(2.0/d1))
            self.biases.append(np.zeros((d2,1)))
            d1 = d2

    def print_model(self):
        """
        This function prints the shapes of weights and biases for each layer.
        """
        print("activation:{}".format(self.activation_function.__class__.__name__))
        print("loss function:{}".format(self.loss_function.__class__.__name__))
        for idx,(w,b) in enumerate(zip(self.weights, self.biases),1):
            print("Layer {}\tw:{}\tb:{}".format(idx, w.shape, b.shape))

    def predict(self, X):
        D = X
        ws = self.weights
        bs = self.biases
        for w,b in zip(ws[:-1], bs[:-1]):
            D = self.activation_function.activate(np.matmul(w,D)+b) 
            # Be careful of the broadcasting here: (d,N) + (d,1) -> (d,N).
        Yhat = np.matmul(ws[-1], D)+bs[-1]
        return np.argmax(Yhat, axis=0)

    def compute_gradients(self, X, Y):
        ws = self.weights
        bs = self.biases
        D_stack = []

        D = X
        D_stack.append(D)
        num_layers = len(ws)
        for idx in range(num_layers-1):
            # TODO 2: Calculate D (D_k in the tutorial) for forward pass (which is similar to self.predit). 
            # This intermediate results to will then be stored to D_stack.

            ### YOUR CODE HERE ###
            raise NotImplementedError("Calculate D")
            D_stack.append(D)

        Yhat = np.matmul(ws[-1], D) + bs[-1]
        training_loss = self.loss_function.loss(Y, Yhat)
        '''
        '''
        grad_bs = []
        grad_Ws = []

        grad = self.loss_function.lossGradient(Y,Yhat)
        grad_b = np.sum(grad, axis=1, keepdims=1)
        grad_W = np.matmul(grad, D_stack[num_layers-1].transpose())
        grad_bs.append(grad_b)
        grad_Ws.append(grad_W)
        for idx in range(num_layers-2, -1, -1):
            # TODO 3: Calculate grad_bs and grad_Ws, which are lists of gradients for b's and w's of each layer. 
            # Take a look at the update step if you are not sure about the format. Notice that we first store the
            # gradients for each layer in a reversed order. The two lists are reversed before returned.

            #1. Update grad for the current layer (G_k in the tutorial)

            ### YOUR CODE HERE ###
            raise NotImplementedError("Update grad")

            #2. Calculate grad_b (gradient with respect to b of the current layer)

            ### YOUR CODE HERE ###
            raise NotImplementedError("Calculate grad_b")
            #3. Calculate grad_W (gradient with respect to W of the current layer)

            ### YOUR CODE HERE ###
            raise NotImplementedError("Calculate grad_W")
            grad_bs.append(grad_b)
            grad_Ws.append(grad_W)

        grad_bs, grad_Ws = grad_bs[::-1], grad_Ws[::-1] # Reverse the gradient lists
        return training_loss, grad_Ws, grad_bs

    def update(self, grad_Ws, grad_bs, learning_rate):
        # Update the weights and biases
        num_layers = len(grad_Ws)
        ws = self.weights
        bs = self.biases
        for idx in range(num_layers):
            ws[idx] -= (grad_Ws[idx] * learning_rate)
            bs[idx] -= (grad_bs[idx] * learning_rate)
        self.weights = ws
        self.biases = bs
        return 

class activationFunction:
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        raise NotImplementedError("Abstract class.")

    def backprop_grad(self, grad):
        """
        The output of backprop_grad should have the same shape as X
        """
        raise NotImplementedError("Abstract class.")

class Relu(activationFunction):
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        return X*(X>0)

    def backprop_grad(self, X):
        """
        The output of backprop_grad should have the same shape as X
        """
        return (X>0).astype(np.float64)

class Linear(activationFunction):
    def activate(self,X):
        """
        The output of activate should have the same shape as X
        """
        return X
    def backprop_grad(self,X):
        """
        The output of backprop_grad should have the same shape as X
        """
        return np.ones(X.shape, dtype=np.float64)

class LossFunction:
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        raise NotImplementedError("Abstract class.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat
        """
        raise NotImplementedError("Abstract class.")

class SquaredLoss(LossFunction):
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        # TODO 0: loss function for squared loss.

        ### YOUR CODE HERE ###
        raise NotImplementedError("Implement SquaredLoss.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat
        """
        #TODO 1: gradient for squared loss.

        ### YOUR CODE HERE ###
        raise NotImplementedError("Implement SquaredLoss.")


class CELoss(LossFunction):
    def loss(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are
        in Yhat; compute the loss associated with these predictions.
        """
        #TODO 4: loss function for cross-entropy loss.

        ### NOT REQUIRED FOR THIS PROJ, YOU CAN DO IT FOR FUN ###
        raise NotImplementedError("Implement CELoss.")

    def lossGradient(self, Y, Yhat):
        """
        The true values are in the vector Y; the predicted values are in 
        Yhat; compute the gradient of the loss with respect to Yhat, which
        has the same shape of Yhat and Y.
        """
        #TODO 5: gradient for cross-entropy loss.

        ### NOT REQUIRED FOR THIS PROJ, YOU CAN DO IT FOR FUN ###
        raise NotImplementedError("Implement CELoss")
