import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dscores = np.zeros_like(X.dot(W))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
    scores = 0.0
    logits = np.dot(X[i], W)
    best_logits = np.exp(logits[y[i]])
   # C = np.exp(-1 * np.max(logits))
    for j in xrange(num_class):
      scores += np.exp(logits[j])
      final_scores = scores
      for j in xrange(num_class):
        dscores[i,j] = np.exp([logits[j]])/final_scores
        if j == y[i]:
          dscores[i,j] -=1
    loss += -np.log(best_logits/ final_scores)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW =X.T.dot(dscores)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  logits = np.dot(X,W)
  num_train = X.shape[0]
  loss = np.sum(-np.log(np.exp(logits[xrange(logits.shape[0]),y])/np.sum(np.exp(logits),axis = 1)))/num_train
  loss += 0.5 * reg * np.sum(W * W)
  dlogits = np.exp(logits)/np.sum(np.exp(logits), axis = 1, keepdims = True)
  dlogits[xrange(logits.shape[0]),y] -= 1
  dlogits /=num_train
  dW = X.T.dot(dlogits) + reg * W
  #loss = np.sum(-y*np.log(np.dot(X,W))-(1-y)*np.log(1-np.log(np.dot(X,W)))) +0.5*reg*np.sum(W*W)
  #dW = reg*W + X.T.dot(np.dot(X,W) - y)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  #dW = np.exp


  return loss, dW
