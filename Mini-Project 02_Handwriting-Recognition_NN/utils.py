
# utils.py
import numpy as np

class tanhActivator(object):
    def forward(self, weighted_input):
        return np.tanh(weighted_input)
    def backward(self, output):
        return 1-output**2
    

