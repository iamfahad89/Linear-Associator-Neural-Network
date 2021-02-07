# Rahaman, Fahad Ur
# 1001-753-107
# 2020-10-11
# Assignment-02-01

import numpy as np
import math

class LinearAssociator(object):
    
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_Limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.number_of_dimensions=input_dimensions
        self.number_of_neurons=number_of_nodes
        self.transfer_function=transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed!=None:
            np.random.seed(seed)
            
        self.random_weights = np.random.randn(self.number_of_neurons, self.number_of_dimensions)

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if W.shape!=self.random_weights.shape:
            return -1
        
        self.random_weights = W

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.random_weights
     
    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        if self.transfer_function=="Hard_limit":
            model_output=np.dot(self.random_weights,X)
            model_output[model_output>=0]=1
            model_output[model_output<0]=0
            return model_output
        else:
            model_output=np.dot(self.random_weights,X)
            return model_output
        
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        pseudo_inverse=np.linalg.pinv(X)
        adjusted_weights=np.dot(y,pseudo_inverse)
        self.random_weights=adjusted_weights
        
    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        e=0
        p=math.ceil(X.shape[1])/batch_size
        
        while (e < num_epochs):
            m=0
            f=0
            e = e + 1
            
            while (f < int(p)):              
                batch_of_inputs=X[:,m:m+batch_size]
                batch_of_inputs_transposed=batch_of_inputs.T
                
                batch_of_outputs=y[:,m:m+batch_size]

                predicted_output=self.predict(batch_of_inputs)
                
                true_output=np.subtract(batch_of_outputs,predicted_output)
                m+=batch_size         
                
                f=f+1
                
                if learning=="Filtered":
                    self.random_weights=(1-gamma)*self.random_weights+np.dot(batch_of_outputs,batch_of_inputs_transposed)*alpha
                    
                elif learning=="delta" or learning=="Delta" or learning=="DeLtA":
                    self.random_weights=self.random_weights+np.dot(true_output,batch_of_inputs_transposed)*alpha

                else:
                    self.random_weights=self.random_weights+np.dot(predicted_output,batch_of_inputs_transposed)*alpha
                
                
    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        pred_value = self.predict(X) 
        mean_squared_error = (np.square(np.subtract(y, pred_value))).mean(None)
        return mean_squared_error
