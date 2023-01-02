
#Section1_neuron.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def load_images():
    images = os.listdir('./data')
    X = []
    Y = np.load('label.npy')
    #load the images
    for i in range(1200):
        #The image is in shape(28,28,3) width=28, height=28, channel(RGB) = 3
        #R=G=B since the figure is gray
        image = cv2.imread('./data/%d.png'%(i+1)).astype('uint8')
        
        #Scale down(up) the figure so that it fit out model(input size: 20*20*1)
        image = cv2.resize(image,(20,20),interpolation=cv2.INTER_AREA)
        X.append(image)
    X = np.array(X)
    return X,Y

def one_hot(label):
    one_hot_label = [0]*26
    one_hot_label[label] = 1
    return one_hot_label    

def load_data():
    X,Y = load_images()
    X = (X.sum(axis=3)//3)/255
    X = X.reshape(X.shape[0],-1) # convert each image into a col vecotr [50*50,1]
    Y = np.array(list(map(one_hot,Y)))
    return X,Y

class tanhActivator(object):
    def forward(self, weighted_input):
        return np.tanh(weighted_input)
    def backward(self, output):
        return 1-output**2

class Neuron():
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = np.array([0.0] * input_num)

    def __str__(self):
        np.set_printoptions(threshold=np.inf)
        return 'w = np.{0}'.format(repr(self.weights))

    def forward(self, x):
        '''
        the forwarding part
        '''
        weighted_sum = np.dot(self.weights.T,x)
        return self.activator.forward(weighted_sum)

    def update_weights(self, x, y, label, learning_rate):
        delta = self.activator.backward(y) * (label - y)
        self.weights = self.weights + learning_rate*delta*x
        
    def train(self, X, labels, epochs=100, learning_rate=0.01):

        for i in range(epochs):
            accuracy = self.one_iteration(X, labels, learning_rate)
            if (i%5 ==0):
                print('epoch %d:'%i,end='')
                print('accuracy in training is %.2f %%'%accuracy)
    def one_iteration(self, X, labels, learning_rate):
        success_case = 0
        for i in range(len(X)):
            x = X[i]
            label = labels[i]
            y = self.forward(x)
            
            if labels[i] == (1 if y>=0.5 else 0):
                success_case += 1
            self.update_weights(x, y, label, learning_rate)
        accuracy = success_case/len(X)*100
        return accuracy


def data_processing():
    def one_hot_to_binary(one_hot_label):
        binary_label = map(lambda x:1 if x[2]==1 else 0,one_hot_label)
        return np.array(list(binary_label)) 
        
    X,Y = load_data()
    train_X,train_Y = X[:80],Y[:80]
    val_X,val_Y = X[80:100],Y[80:100]
    train_Y = one_hot_to_binary(train_Y)
    val_Y = one_hot_to_binary(val_Y)
    #amount_of_c_in_train = train_Y.sum()
    #amount_of_c_in_val =val_Y.sum()
    #print('amount_of_c_in_train=%d'%amount_of_c_in_train)
    #print('amount_of_c_in_val=%d'%amount_of_c_in_val)
    return train_X,train_Y,val_X,val_Y
    

if __name__ == '__main__':
    train_X,train_Y,val_X,val_Y = data_processing()
    neuron = Neuron(train_X.shape[1],tanhActivator())
    #training
    neuron.train(train_X,train_Y,100,0.01)
    
    #validation
    success_case = 0
    for i in range(len(val_X)):
        #calssify the image is c if the output >=0.5
        if val_Y[i] == (1 if neuron.forward(val_X[i])>=0.5 else 0):
            success_case += 1
    accuracy = success_case/len(val_X)*100
    print('accuarcy in validation dataset is %.2f %%'%accuracy)
    

