"""
Implements a two-layer Neural Network classifier in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import random
import statistics
from linear_classifier import sample_batch


def hello_two_layer_net():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from two_layer_net.py!')


# Template class modules that we will use later: Do not edit/modify this class
class TwoLayerNet(object):
  def __init__(self, input_size, hidden_size, output_size,
               dtype=torch.float32, device='cuda', std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    - dtype: Optional, data type of each initial weight params
    - device: Optional, whether the weight params is on GPU or CPU
    - std: Optional, initial weight scaler.
    """
    # reset seed before start
    random.seed(0)
    torch.manual_seed(0)

    self.params = {}
    self.params['W1'] = std * torch.randn(input_size, hidden_size, dtype=dtype, device=device)
    self.params['b1'] = torch.zeros(hidden_size, dtype=dtype, device=device)
    self.params['W2'] = std * torch.randn(hidden_size, output_size, dtype=dtype, device=device)
    self.params['b2'] = torch.zeros(output_size, dtype=dtype, device=device)

  def loss(self, X, y=None, reg=0.0):
    return nn_forward_backward(self.params, X, y, reg)

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    return nn_train(
            self.params,
            nn_forward_backward,
            nn_predict,
            X, y, X_val, y_val,
            learning_rate, learning_rate_decay,
            reg, num_iters, batch_size, verbose)

  def predict(self, X):
    return nn_predict(self.params, nn_forward_backward, X)

  def save(self, path):
    torch.save(self.params, path)
    print("Saved in {}".format(path))

  def load(self, path):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint
    print("load checkpoint file: {}".format(path))



def nn_forward_pass(params, X):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:

    FC layer -> ReLU (hidden) -> FC layer (scores)

    As a practice, we will NOT allow to use torch.relu and torch.nn ops
    just for this time (you can use it from A3).

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None
    ############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input.#
    # Store the result in the scores variable, which should be an tensor of    #
    # shape (N, C).                                                            #
    ############################################################################
    # Replace "pass" statement with your code
 
    hidden = torch.mm(X, W1) + b1   #(N,D)(D,H) + (H,)  --> (N,H)
    hidden[hidden < 0] = 0           #(N,H)
    scores = torch.mm(hidden, W2) + b2  #(N,H)(H,C) + (C,) ---> (N,C)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return scores, hidden


def nn_forward_backward(params, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network. When you implement loss and gradient, please don't forget to
    scale the losses/gradients by the batch size.

    Inputs: First two parameters (params, X) are same as nn_forward_pass
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    scores, h1 = nn_forward_pass(params, X)
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    ############################################################################
    # TODO: Compute the loss, based on the results from nn_forward_pass.       #
    # This should include both the data loss and L2 regularization for W1 and  #
    # W2. Store the result in the variable loss, which should be a scalar. Use #
    # the Softmax classifier loss. When you implment the regularization over W,#
    # please DO NOT multiply the regularization term by 1/2 (no coefficient).  #
    # If you are not careful here, it is easy to run into numeric instability  #
    # (Check Numeric Stability in http://cs231n.github.io/linear-classify/).   #
    ############################################################################
    # Replace "pass" statement with your code

    scores -= torch.max(scores, dim=1, keepdim=True)[0]  #(N,C)
    prob = torch.exp(scores)   #(N,C)
    prob /= torch.sum(prob, dim=1, keepdim=True) #(N,C)
    loss = torch.sum(-torch.log(prob[range(0, N), y])).item() / N
    loss += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
    


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    # Backward pass: compute gradients
    grads = {}
    ###########################################################################
    # TODO: Compute the backward pass, computing the derivatives of the       #
    # weights and biases. Store the results in the grads dictionary.          #
    # For example, grads['W1'] should store the gradient on W1, and be a      #
    # tensor of same size                                                     #
    ###########################################################################
    # Replace "pass" statement with your code

    dscores = prob  # (N,C) 
    dscores[range(0, N), y] -= 1  
    dscores /= N
    # W2,b2
    grads['W2'] = torch.mm(h1.t(), dscores)  #(H,N)(N,C)
    grads['b2'] = torch.sum(dscores, dim=0)  
    dhidden = torch.mm(dscores, W2.t())  #(N,C)(C,H)
    dhidden[h1 <= 0] = 0 
    grads['W1'] = torch.mm(X.t(), dhidden)   #(D,N)(N,H)  --->D,H
    grads['b1'] = torch.sum(dhidden, dim=0)
    # Regularization component
    grads['W2'] += 2 * reg * W2
    grads['W1'] += 2 * reg * W1



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, grads


def nn_train(params, loss_func, pred_func, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
  """
  Train this neural network using stochastic gradient descent.

  Inputs:
  - params: a dictionary of PyTorch Tensor that store the weights of a model.
    It should have following keys with shape
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
  - loss_func: a loss function that computes the loss and the gradients.
    It takes as input:
    - params: Same as input to nn_train
    - X_batch: A minibatch of inputs of shape (B, D)
    - y_batch: Ground-truth labels for X_batch
    - reg: Same as input to nn_train
    And it returns a tuple of:
      - loss: Scalar giving the loss on the minibatch
      - grads: Dictionary mapping parameter names to gradients of the loss with
        respect to the corresponding parameter.
  - pred_func: prediction function that im
  - X: A PyTorch tensor of shape (N, D) giving training data.
  - y: A PyTorch tensor f shape (N,) giving training labels; y[i] = c means that
    X[i] has label c, where 0 <= c < C.
  - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
  - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
  - learning_rate: Scalar giving learning rate for optimization.
  - learning_rate_decay: Scalar giving factor used to decay the learning rate
    after each epoch.
  - reg: Scalar giving regularization strength.
  - num_iters: Number of steps to take when optimizing.
  - batch_size: Number of training examples to use per step.
  - verbose: boolean; if true print progress during optimization.

  Returns: A dictionary giving statistics about the training process
  """
  num_train = X.shape[0]
  iterations_per_epoch = max(num_train // batch_size, 1)

  # Use SGD to optimize the parameters in self.model
  loss_history = []
  train_acc_history = []
  val_acc_history = []

  for it in range(num_iters):
    X_batch, y_batch = sample_batch(X, y, num_train, batch_size)

    # Compute loss and gradients using the current minibatch
    loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
    loss_history.append(loss.item())

    #########################################################################
    # TODO: Use the gradients in the grads dictionary to update the         #
    # parameters of the network (stored in the dictionary self.params)      #
    # using stochastic gradient descent. You'll need to use the gradients   #
    # stored in the grads dictionary defined above.                         #
    #########################################################################
    # Replace "pass" statement with your code
    params['W1'] -= learning_rate * grads['W1'] 
    params['W2'] -= learning_rate * grads['W2']
    params['b1'] -= learning_rate * grads['b1']
    params['b2'] -= learning_rate * grads['b2']
    #########################################################################
    #                             END OF YOUR CODE                          #
    #########################################################################

    if verbose and it % 100 == 0:
      print('iteration %d / %d: loss %f' % (it, num_iters, loss.item()))

    # Every epoch, check train and val accuracy and decay learning rate.
    if it % iterations_per_epoch == 0:
      # Check accuracy
      y_train_pred = pred_func(params, loss_func, X_batch)
      train_acc = (y_train_pred == y_batch).float().mean().item()
      y_val_pred = pred_func(params, loss_func, X_val)
      val_acc = (y_val_pred == y_val).float().mean().item()
      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      # Decay learning rate
      learning_rate *= learning_rate_decay

  return {
    'loss_history': loss_history,
    'train_acc_history': train_acc_history,
    'val_acc_history': val_acc_history,
  }


def nn_predict(params, loss_func, X):
  """
  Use the trained weights of this two-layer network to predict labels for
  data points. For each data point we predict scores for each of the C
  classes, and assign each data point to the class with the highest score.

  Inputs:
  - params: a dictionary of PyTorch Tensor that store the weights of a model.
    It should have following keys with shape
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
  - loss_func: a loss function that computes the loss and the gradients
  - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
    classify.

  Returns:
  - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
    the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
    to have class c, where 0 <= c < C.
  """
  y_pred = None

  ###########################################################################
  # TODO: Implement this function; it should be VERY simple!                #
  ###########################################################################
  # Replace "pass" statement with your code
  fc1 = torch.mm(X, params['W1']) + params['b1']
  fc1[fc1 < 0] = 0
  scores = torch.mm(fc1, params['W2']) + params['b2']
  _, y_pred = torch.max(scores, dim=1)

  ###########################################################################
  #                              END OF YOUR CODE                           #
  ###########################################################################

  return y_pred



def nn_get_search_params():
  """
  Return candidate hyperparameters for a TwoLayerNet model.
  You should provide at least two param for each, and total grid search
  combinations should be less than 256. If not, it will take
  too much time to train on such hyperparameter combinations.

  Returns:
  - learning_rates: learning rate candidates, e.g. [1e-3, 1e-2, ...]
  - hidden_sizes: hidden value sizes, e.g. [8, 16, ...]
  - regularization_strengths: regularization strengths candidates
                              e.g. [1e0, 1e1, ...]
  - learning_rate_decays: learning rate decay candidates
                              e.g. [1.0, 0.95, ...]
  """
  learning_rates = []
  hidden_sizes = []
  regularization_strengths = []
  learning_rate_decays = []
  ###########################################################################
  # TODO: Add your own hyper parameter lists. This should be similar to the #
  # hyperparameters that you used for the SVM, but you may need to select   #
  # different hyperparameters to achieve good performance with the softmax  #
  # classifier.                                                             #
  ###########################################################################
  # Replace "pass" statement with your code
  learning_rates = [0.6, 0.8, 1]
  hidden_sizes = [32, 64, 128]
  regularization_strengths = [0.0006, 0.0001]
  learning_rate_decays = [0.95, 0.90, 0.85]
  ###########################################################################
  #                           END OF YOUR CODE                              #
  ###########################################################################

  return learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays


def find_best_net(data_dict, get_param_set_fn):
  """
  Tune hyperparameters using the validation set.
  Store your best trained TwoLayerNet model in best_net, with the return value
  of ".train()" operation in best_stat and the validation accuracy of the
  trained best model in best_val_acc. Your hyperparameters should be received
  from in nn_get_search_params

  Inputs:
  - data_dict (dict): a dictionary that includes
                      ['X_train', 'y_train', 'X_val', 'y_val']
                      as the keys for training a classifier
  - get_param_set_fn (function): A function that provides the hyperparameters
                                 (e.g., nn_get_search_params)
                                 that gives (learning_rates, hidden_sizes,
                                 regularization_strengths, learning_rate_decays)
                                 You should get hyperparameters from
                                 get_param_set_fn.

  Returns:
  - best_net (instance): a trained TwoLayerNet instances with
                         (['X_train', 'y_train'], batch_size, learning_rate,
                         learning_rate_decay, reg)
                         for num_iter times.
  - best_stat (dict): return value of "best_net.train()" operation
  - best_val_acc (float): validation accuracy of the best_net
  """

  best_net = None
  best_stat = None
  best_val_acc = 0.0

  #############################################################################
  # TODO: Tune hyperparameters using the validation set. Store your best      #
  # trained model in best_net.                                                #
  #                                                                           #
  # To help debug your network, it may help to use visualizations similar to  #
  # the ones we used above; these visualizations will have significant        #
  # qualitative differences from the ones we saw above for the poorly tuned   #
  # network.                                                                  #
  #                                                                           #
  # Tweaking hyperparameters by hand can be fun, but you might find it useful #
  # to write code to sweep through possible combinations of hyperparameters   #
  # automatically like we did on the previous exercises.                      #
  #############################################################################
#   # Replace "pass" statement with your code
  learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays = nn_get_search_params()
  num_models = len(learning_rates) * len(regularization_strengths) * len(hidden_sizes) * len(learning_rate_decays)
  input_size = 3 * 32 * 32
  num_classes = 10
  i = 0
  # results is dictionary mapping tuples of the form
  # (learning_rate, hidden_sizes, regularization_strengths, learning_rate_decays) to tuples of the form
  # (train_acc, val_acc). 
  results = {}
  
  
  num_iters = 100 # number of iterations

  for lr in learning_rates:
    for hs in hidden_sizes:
      for rs in regularization_strengths:
        for lrd in learning_rate_decays:
            i += 1
            print("Traning net with learning_rate={}, hidden_size={}, regularization_strength={}, learning_rate_deacys={}"\
                .format(lr,hs,rs,lrd))
    
            
            net = TwoLayerNet(input_size, hs, num_classes, dtype=data_dict['X_train'].dtype, device=data_dict['X_train'].device)
          
            # Train the network
            stats = net.train(data_dict['X_train'], data_dict['y_train'],
                              data_dict['X_val'], data_dict['y_val'],
                              num_iters=3000, batch_size=1000,
                              learning_rate=lr, learning_rate_decay=lrd,
                              reg=rs, verbose=True)

            # Predict on the Train set
            y_val_pred = net.predict(data_dict['X_train'])
            cand_train_acc = 100.0 * (y_val_pred == data_dict['y_train']).double().mean().item()

            # Predict on the validation set
            y_val_pred = net.predict(data_dict['X_val'])
            cand_val_acc = 100.0 * (y_val_pred == data_dict['y_val']).double().mean().item()

    

            if cand_val_acc > best_val_acc:
              best_val_acc = cand_val_acc
              best_net = net # save the svm
              best_stat = stats
              results[(lr, hs, rs, lrd)] = (cand_train_acc, cand_val_acc)

  for lr, hs, rs, lrd in sorted(results):
    train_acc, val_acc = results[(lr, hs, rs, lrd)]
    print("learning_rate={}, hidden_size={}, regularization_strength={}, learning_rate_deacys={} : Train acc={}, Val acc ={}"\
                .format(lr, hs, rs, lrd, train_acc, val_acc))




  
  #############################################################################
  #                               END OF YOUR CODE                            #
  #############################################################################

  return best_net, best_stat, best_val_acc