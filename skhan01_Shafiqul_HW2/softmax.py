import numpy as np


class Softmax(object):
    """" Softmax classifier """

    def __init__(self, inputDim: object, outputDim: object) -> object:
        self.W = None
        #########################################################################
        # TODO: 5 points                                                        #
        # - Generate a random softmax weight matrix to use to compute loss.     #
        #   with standard normal distribution and Standard deviation = 0.01.    #
        #########################################################################

        self.inputDim = inputDim

        print(inputDim)

        self.outputDim = outputDim

        print (outputDim)

        self.W = 0.01 * np.random.randn(inputDim, outputDim)



        pass
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss(self, x, y, reg):
        """
        Softmax loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to weights self.W (dW) with the same shape of self.W.
        """
        loss = 0.0
        dW = np.zeros_like(self.W)

        #############################################################################
        # TODO: 20 points                                                           #
        # - Compute the softmax loss and store to loss variable.                    #
        # - Compute gradient and store to dW variable.                              #
        # - Use L2 regularization                                                  #
        # Bonus:                                                                    #
        # - +2 points if done without loop                                          #
        #############################################################################

        N = x.shape[0]
        C = self.W.shape[1]

        s = np.dot(x, self.W)
        s -= np.max(s,axis=1).reshape(N,1)
       # maximum_row = np.max(s, axis=1)
       # s -= maximum_row.reshape((N, 1))
        bottom = np.sum(np.exp(s), axis=1)
        top = np.exp(np.choose(y, s.T))
        #s_C = np.choose(y, s.T)
        #top = np.exp(s_C)

        prob = np.exp(s) / np.sum(np.exp(s), axis=1).reshape(N, 1)
        prob_y = np.zeros((N, C))
        prob_y[np.arange(prob_y.shape[0]), y] = 1
        delta = prob - prob_y
        dW = np.dot(x.T, delta)
        loss = np.sum(-np.log(top / bottom), axis=0)
        loss /= N
        loss += 0.5 * reg * np.sum(self.W ** 2)
        dW /= N
        dW += reg * self.W



        pass
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def train(self, x, y, lr=1e-3, reg=1e-5, iter=100, batchSize=200, verbose=False):
        """
        Train this Softmax classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iter):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (D, batchSize)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            training_data = np.random.choice(x.shape[0], batchSize, replace=True)
            xBatch = x[training_data]
            yBatch = y[training_data]
            loss, dW = self.calLoss(xBatch, yBatch, reg)
            lossHistory.append(loss)

            self.W -= lr * dW

            pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict(self, x):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])

        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        Y_prob = x.dot(self.W)

        yPred = np.argmax(Y_prob, axis=1)
        print(yPred)

        pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred

    def calAccuracy(self, x, y):
        acc = 0
        N = x.shape[0]
        yPred =0
        Y_prob = x.dot(self.W)

        yPred = np.argmax(Y_prob, axis=1)
        ###########################################################################
        # TODO: 5 points                                                          #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        acc = np.mean(yPred == y)

        pass

          ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################


        return acc