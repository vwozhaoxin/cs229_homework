import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  margins = 0.0
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  #loss_delta = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      #if y[i] * scores[j] <1:
      #  dW[i,j]= -y[i] * X[i,j]
     # W_delta = W
     # W_delta[i,j] += 0.0001
      #scores_delta = X[i].dot(W_delta)
     # margin_delta = scores_delta[j] - scores_delta[y[i]] + 1
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1 KEEP OTHER CLASS IS 0 AND THE RIGHT CLASS IS 1
      margins +=margin
      if margin > 0:
        loss += margin
        dW[:,j] +=X[i]
        dW[:,y[i]] -= X[i]
    #    loss_delta +=margin_delta
     # if margin >0 and margin_delta >0:
      #  dW[i, j] = X[i, j] - y[i] * X[i, j]
        #dW[i, j] = (margin_delta - margin) / 0.0001

  #print(margins)
  #print('-'*20)

        #dW[i, j] = X[i, j]* W[i,j]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  #loss_delta /=num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W) *0.5
  dW += reg * W
 # loss_delta += reg*np.sum(W*W)

  #dW = np.mean(np.sum(dW, axis=0, keepdims=True), keepdims=True)
  #dW = np.sum(dW, axis = 0, keepdims = True)
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  #loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #num_classes = W.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  scores = np.dot(X,W)
  #print(scores.shape)
  #print(scores[y].shape)
  best_scores = scores[range(scores.shape[0]),y]
  best_scores = best_scores.reshape(-1, 1)
  margins = np.maximum(0, scores - best_scores + 1)
  margins[range(margins.shape[0]),y]= 0

 # print(np.sum(margins))
  loss = np.sum(margins)/margins.shape[0] + reg * np.sum(W * W)*0.5
  '''
  scores = np.dot(X, W)  # also known as f(x_i, W)

  correct_scores = np.ones(scores.shape) * scores[y]
  deltas = np.ones(scores.shape)
  L = scores - correct_scores + deltas

  L[L < 0] = 0
  L[np.arange(0, scores.shape[0]),y] = 0  # Don't count y_i
  loss = np.sum(L)

  # Average over number of training examples
  num_train = X.shape[0]
  loss /= num_train

  # Add regularization
  loss += 0.5 * reg * np.sum(W * W)
  '''
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #print(margins[margins>0].reshape((X.shape[0],W.shape[1])))

  #print(non_zeros_count.shape)

  margins[margins>0] =1
  #print(margins)
  margins[range(margins.shape[0]),y] = -np.sum(margins,axis=1)
  #print(dW.shape)
  dW = X.T.dot(margins) / X.shape[0]
  dW = dW + reg*W
  #dW=X.T.dot(margins[margins>0].reshape(-1,W.shape[1])) + reg * W
 #dW[xrange(scores)]
  assert dW.shape == W.shape
  '''

  grad = np.zeros(scores.shape)

  L = scores - correct_scores + deltas

  L[L < 0] = 0
  L[L > 0] = 1
  #L[np.arange(0, scores.shape[0]),y] = 0 # Don't count y_i
  L[np.arange(0, scores.shape[0]),y] = -1 * np.sum(L, axis=1)
  dW = np.dot(X.T,L)

  # Average over number of training examples
  num_train = X.shape[0]
  dW /= num_train
  dW += reg *W
  '''
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
