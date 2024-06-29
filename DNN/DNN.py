import argparse
import sys
import numpy as np

class neuralNet:
    
    def __init__(self, args):
        self.verbose = args.v
        self.train_feat_fn = args.TRAIN_FEAT_FN        
        self.train_target_fn = args.TRAIN_TARGET_FN        
        self.dev_feat_fn = args.DEV_FEAT_FN        
        self.dev_target_fn = args.DEV_TARGET_FN      
        self.epochs = args.EPOCHS       
        self.learnrate = args.LEARNRATE        
        self.num_hidden_units = args.NUM_HIDDEN_UNITS      
        self.problem_mode = args.PROBLEM_MODE        
        self.hidden_unit_activation = args.HIDDEN_UNIT_ACTIVATION       
        self.init_range = args.INIT_RANGE        
        self.c = args.C        
        self.minibatch_size = args.MINIBATCH_SIZE        
        self.num_hidden_layers = args.NUM_HIDDEN_LAYERS    
    
    # activation functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def relu_deriv(self, Z):
        return np.where(Z > 0, 1, 0)
    
    def sigmoid_deriv(self, Z):
        sig = self.sigmoid(Z)
        return sig * (1 - sig)

    def tanh_deriv(self, x):
        return 1 - np.tanh(x)**2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    # one-hot encoding for classification
    def one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
        
        return one_hot    
    
    def load_data(self, t_feat, t_target, d_feat, d_target):
        # load datasets specified in cmd line args
        train_features = np.loadtxt(t_feat)
        train_target = np.loadtxt(t_target)
        dev_features = np.loadtxt(d_feat)
        dev_target = np.loadtxt(d_target)

        # One-hot encode the targets if the problem mode is 'C' for classification
        if self.problem_mode == 'C':
            train_target = self.one_hot_encode(train_target, self.c)
            dev_target = self.one_hot_encode(dev_target, self.c)

        return train_features, train_target, dev_features, dev_target


    # initialize model parameters (some taken from cmd line args)
    def init_params(self, num_features, output_size):
        params = {}

        # Input layer size + hidden layer sizes + output layer size
        layer_sizes = [num_features] + [self.num_hidden_units] * self.num_hidden_layers + [output_size]

        # Initialize weights and biases for each layer
        for i in range(1, len(layer_sizes)):
            params['W' + str(i)] = np.random.uniform(-self.init_range, self.init_range, (layer_sizes[i], layer_sizes[i-1]))
            params['b' + str(i)] = np.zeros((layer_sizes[i], 1))  # It's common to initialize biases to zeros
        return params

    # forward pass
    def forward_prop(self, inputs, params, activation_func):
        caches = []
        A = inputs.T        
        caches.append((inputs, A))

        # loop through hidden layers
        for i in range(1, self.num_hidden_layers + 1):
            # get params
            W = params['W' + str(i)]
            b = params['b' + str(i)]

            # linear transformation (mm of wight and activation + bias)
            Z = np.dot(W, A) + b

            # apply activation function
            if activation_func == 'relu':
                A = self.relu(Z)
            elif activation_func == 'sig':
                A = self.sigmoid(Z)
            elif activation_func == 'tanh':
                A = self.tanh(Z)
            else:
                raise ValueError('Unknown activation function "{}"'.format(self.hidden_unit_activation))

            # store values for backprop
            cache = (Z, A)
            caches.append(cache)

        # output layer
        W_last = params['W' + str(self.num_hidden_layers+1)]    
        b_last = params['b' + str(self.num_hidden_layers+1)]       
        Z_last = np.dot(W_last, A) + b_last#.reshape(1, -1)

        if self.problem_mode == 'C':
            A_last = self.softmax(Z_last.T)
        else:
            A_last = Z_last.T  # take the linear output
        cache = (Z_last, A_last)
        caches.append(cache)

        return A_last, caches
        

    # compute accuracy for classification tasks
    def compute_acc_classif(self, softmax_output, y_mini):
        predictions = np.argmax(softmax_output, axis=1)
        targets = np.argmax(y_mini, axis=1)
        acc = np.mean(predictions == targets)

        return acc

    # compute error for regression tasks
    def compute_err_reg(self, Y_true, Y_pred):
        # mean squared error
        mse = np.square(np.subtract(Y_true, Y_pred)).mean()
        return mse

    # backward pass
    def back_prop(self, A_last, Y, caches, params):
        gradients = {}
        m = Y.shape[0]
        L = len(caches)
        Y = Y.reshape(A_last.shape)
        
        dA = A_last - Y
        for l in reversed(range(1, len(caches))):
            # Retrieve current layer's cache and unpack Z and A
            current_cache = caches[l]
            Z_current, A_current = current_cache
            
            # Gradient of the loss with respect to Z (dZ)
            if l == len(caches) - 1: 
                dZ = dA 
            else:
                # Apply derivative of activation function
                if self.hidden_unit_activation == 'relu':
                    dZ = dA * self.relu_deriv(Z_current.T)
                elif self.hidden_unit_activation == 'sig':
                    dZ = dA * self.sigmoid_deriv(Z_current.T)
                elif self.hidden_unit_activation == 'tanh':
                    dZ = dA * self.tanh_deriv(Z_current.T)
                else:
                    raise ValueError('Unknown activation function "{}"'.format(self.hidden_unit_activation))

            # Retrieve the previous layer's activation
            A_prev = caches[l-1][1]
            
            # Calculate gradients
            gradients['dW' + str(l)] = (1 / m) * np.dot(dZ.T, A_prev.T)
            gradients['db' + str(l)] = (1 / m) * np.sum(dZ.T, axis=1, keepdims=True)
            
            # Not the input layer, propagate the gradient backwards
            if l > 1:
                # Gradient of loss with respect to activation of previous layer
                dA = np.dot(params['W' + str(l)].T, dZ.T).T  
        
        return gradients

    # update weights and biases
    def update_params(self, gradients, params):
        # basic gradient descent
        for layer in range(1, self.num_hidden_layers + 2):
            params['W' + str(layer)] -= self.learnrate * gradients['dW' + str(layer)]
            params['b' + str(layer)] -= self.learnrate * gradients['db' + str(layer)]

    # logic for classification mode
    def classification(self, mini_batches, epochs, params, x_dev, y_dev, x_train, y_train):
        updates = 0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(epochs):
            if self.minibatch_size == 0:
                # full batch training
                softmax_output, caches = self.forward_prop(x_train, params, self.hidden_unit_activation)
                gradients = self.back_prop(softmax_output, y_train, caches, params)
                self.update_params(gradients, params)
                acc_train = self.compute_acc_classif(softmax_output, y_train)
            
            else:
                # mini-batch training
                for mini_batch in mini_batches:
                    updates += 1
                    x_mini, y_mini = mini_batch 
                    # forward pass
                    softmax_output, caches = self.forward_prop(x_mini, params, self.hidden_unit_activation)
                    
                    # backward pass
                    gradients = self.back_prop(softmax_output, y_mini, caches, params)
                    
                    # update params
                    self.update_params(gradients, params)

                    if self.verbose:
                        # compute accuracy
                        acc_train = self.compute_acc_classif(softmax_output, y_mini)

                        # evaluate on dev set
                        acc_dev = self.evaluate(x_dev, y_dev, params)                    

                        print(f"Update {updates:06d}: train={acc_train:.3f} dev={acc_dev:.3f}", file=sys.stderr)
                    
                    else:
                        # compute accuracy for non-verbose mode
                        predictions = np.argmax(softmax_output, axis=1)
                        targets = np.argmax(y_mini, axis=1)
                        correct_predictions += np.sum(predictions == targets)
                        total_predictions += targets.shape[0]

                if not self.verbose:
                    # compute accuracy
                    acc_train = correct_predictions / total_predictions                

            # evaluate on dev set
            acc_dev = self.evaluate(x_dev, y_dev, params)

            print(f"Epoch {epoch:03d}: train={acc_train:.3f} dev={acc_dev:.3f}", file=sys.stderr)
    
    # logic for regression mode
    def regression(self, mini_batches, x_train, epochs, params, x_dev, y_dev):
        for epoch in range(epochs):
            mse_epoch = 0
            
            for x_mini, y_mini in mini_batches:
                # forward pass
                predictions, caches = self.forward_prop(x_mini, params, self.hidden_unit_activation)
                
                # compute loss (mean squared error)
                mse_batch = self.compute_err_reg(predictions, y_mini)
                mse_epoch += mse_batch * x_mini.shape[0]
                
                # backward pass
                gradients = self.back_prop(predictions, y_mini, caches, params)
                
                # update params
                self.update_params(gradients, params)
                
                if self.verbose:
                    # evaluate on dev set
                    mse_eval = self.evaluate(x_dev, y_dev, params)
                    print(f"Update {epoch * len(mini_batches):06d}: train={mse_batch:.3f} dev={mse_eval:.3f}", file=sys.stderr)

            mse_train = mse_epoch / x_train.shape[0]

            # evaluate on dev set
            mse_dev = self.evaluate(x_dev, y_dev, params)

            print(f"Epoch {epoch:03d}: train={mse_train:.3f} dev={mse_dev:.3f}", file=sys.stderr)

    # Train the model
    def train(self, x_train, y_train, epochs, x_dev, y_dev):
        # initialize params
        params = self.init_params(x_train.shape[1], self.c)
        
        if self.problem_mode == 'C':
            if self.minibatch_size == 0:
                self.classification(0, epochs, params, x_dev, y_dev, x_train, y_train)
            else:
                mini_batch_size = self.minibatch_size
                mini_batches = self.create_mini_batches(x_train, y_train, mini_batch_size)                               
                self.classification(mini_batches, epochs, params, x_dev, y_dev, x_train, y_train)
        
        elif self.problem_mode == 'R':
            if self.minibatch_size == 0:
                self.regression(0, x_train, epochs, params, x_dev, y_dev)
            else:
                mini_batch_size = self.minibatch_size
                mini_batches = self.create_mini_batches(x_train, y_train, mini_batch_size)
                self.regression(mini_batches, epochs, params, x_dev, y_dev)

    # evaluate the model on validation set
    def evaluate(self, X_eval, Y_eval, params):
        # forward pass to get predictions
        predictions, _ = self.forward_prop(X_eval, params, self.hidden_unit_activation)

        # For classification, compute accuracy
        if self.problem_mode == 'C':
            # convert predicted probabilities to class labels
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(Y_eval, axis=1)
            accuracy = np.mean(predicted_classes == true_classes)
            return accuracy
        # compute mse for regression
        elif self.problem_mode == 'R':            
            mse = self.compute_err_reg(Y_eval, predictions)
            return mse
        
    # Create mini-batches from dataset
    def create_mini_batches(self, x, y, batch_size):
        mini_batches = []
        data = np.hstack((x, y))
        np.random.shuffle(data) # shuffle data
        n_minibatches = data.shape[0] // batch_size

        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
            # Separate mini-batch data into features and targets
            X_mini = mini_batch[:, :-self.c] 
            Y_mini = mini_batch[:, -self.c:] 
            mini_batches.append((X_mini, Y_mini))  
        # Check if there's a last mini-batch with fewer than 'batch_size' examples          
        if data.shape[0] % batch_size != 0:
            mini_batch = data[-(data.shape[0] % batch_size):]
            X_mini = mini_batch[:, :-self.c]
            Y_mini = mini_batch[:, -self.c:]
            mini_batches.append((X_mini, Y_mini))

        return mini_batches

# parse command line arguments
def cmd_line_parser():
    parser = argparse.ArgumentParser(description="Neural Network args")
    parser.add_argument('-v', action='store_true', help="enable verbose mode")
    parser.add_argument('TRAIN_FEAT_FN', help="name of training set feature file")        
    parser.add_argument('TRAIN_TARGET_FN', help="name of training set target (label) file")
    parser.add_argument('DEV_FEAT_FN', help="name of the development set feature file")
    parser.add_argument('DEV_TARGET_FN', help="name of the development set target (label) file")
    parser.add_argument('EPOCHS', type=int, help="total number of epochs to train for")
    parser.add_argument('LEARNRATE', type=float, help="step size to use for training (MB-SGD)")
    parser.add_argument('NUM_HIDDEN_UNITS', type=int, help="dimension of the hidden layers")
    parser.add_argument('PROBLEM_MODE', help="should be either C or R for classification or regression")
    parser.add_argument('HIDDEN_UNIT_ACTIVATION', help="the element-wise, non-linear function to apply at each hidden layer")
    parser.add_argument('INIT_RANGE', type=float, help="all the weights should be initialized uniformly random in this range")
    parser.add_argument('C', type=int, help="number of classes (classification) or dimension of output vector (regression)")
    parser.add_argument('MINIBATCH_SIZE', type=int, help="num data points to be included in each mini-batch")
    parser.add_argument('NUM_HIDDEN_LAYERS', type=int, help="num of hidden layers in NN")
    return parser.parse_args()
    
def main():
    args = cmd_line_parser()
    net = neuralNet(args)
    t_feat, t_targ, d_feat, d_targ = net.load_data(net.train_feat_fn, net.train_target_fn,
                                                            net.dev_feat_fn, net.dev_target_fn)

    # train the model
    net.train(t_feat, t_targ, net.epochs, d_feat, d_targ)

if __name__ == "__main__":
    main()