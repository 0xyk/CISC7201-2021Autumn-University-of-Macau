
#layer.py

import numpy as np
class FullConnectedLayer():
    def __init__(self, input_size, output_size, 
                 activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        self.y = np.zeros((output_size, 1))
    def forward(self, x):
        self.x = x
        self.y = self.activator.forward(
            np.dot(self.W, x) + self.b)
            
    def backward(self, delta_array):
        '''
        Your task 1
        - calculate the Gradient in Eq.1
        - (refer to the 2nd term in Eq.1)
        '''
        
        # ****** Your code begin (task 1) ******
        # self.W_grad = the dot product of delta_array and the transpose of self.x
        self.W_grad = np.dot(delta_array,self.x.T)
        self.b_grad = delta_array        
        # ****** Your code end (task 1) ******
        
        # Don't touch other code
        self.delta = self.activator.backward(self.x) * np.dot(
            self.W.T, delta_array)
    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

