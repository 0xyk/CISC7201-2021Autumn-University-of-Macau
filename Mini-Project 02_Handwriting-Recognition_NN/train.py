
#train.py
#Data Augmentation + 5 layerNN + bias + tanhActivator + dynamic learning rate + weights precision is 24
#All code here

#data.py
# Used to load images
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
# load the images

def load_images():
    images = os.listdir('./data')
    X = []
    Y = np.load('label.npy')
    #load the images
    for i in range(1200):
        #The image is in shape(28,28,3) width=28, height=28, channel(RGB) = 3
        #R=G=B since the figure is gray
        image = cv2.imread('data/%d.png'%(i+1)).astype('uint8')
        
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

# utils.py
# define the sigmoid activator and it's gradient (derivative) for backpropagation
class tanhActivator(object):
    def forward(self, weighted_input):
        return np.tanh(weighted_input)    
    def backward(self, output):
        return 1-output**2
    
#layer.py

class FullConnectedLayer():
    def __init__(self, input_size, output_size, 
                 activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1,
            (output_size, input_size))
        
        self.b = np.random.uniform(-0.1,0.1,(output_size, 1))

        
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
        # Eq.1 in Sec. 2.4
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad
        
#3_2layerNN.py

class Network():
    def __init__(self, input_size,hidden_size,output_size):
        self.layers = []
        self.layers.append(FullConnectedLayer(
                        input_size, hidden_size,
                        tanhActivator()
                        )
                    )
        self.layers.append(FullConnectedLayer(
                        hidden_size, hidden_size,
                        tanhActivator()
                        )
                    )
        self.layers.append(FullConnectedLayer(
                        hidden_size, hidden_size,
                        tanhActivator()
                        )
                    )
        self.layers.append(FullConnectedLayer(
                        hidden_size, hidden_size,
                        tanhActivator()
                        )
                    )
        self.layers.append(FullConnectedLayer(
                        hidden_size, output_size,
                        tanhActivator()
                        )
                    )
    def __str__(self):
        np.set_printoptions(threshold=np.inf)
        result = ''
        for i in range(len(self.layers)):
            result +='W{1} = np.{0}'.format(repr(self.layers[i].W),i+1) +  '\n\n' 
        
        for i in range(len(self.layers)):
            result +='b{1} = np.{0}'.format(repr(self.layers[i].b),i+1) +  '\n\n' 

        return result
    
    def train(self, X, Y, epoch=100,learning_rate=0.01):
        for i in range(epoch):
            success_case = 0
            if epoch == 200:
                learning_rate /=10
            if epoch == 500:
                learning_rate /= 10
            for d in range(len(X)):
                y = self.train_one_sample(X[d], 
                    Y[d], learning_rate)
                if np.argmax(y) == np.argmax(Y[d]):
                    success_case+=1
                    
            #validation
            if (i%5 ==0):
                success_case_val = 0
                for d in range(len(val_X)):
                    #calssify the image is c if the output >=0.5
                    if np.argmax(val_Y[d]) == np.argmax(model.forward(val_X[d])):
                        success_case_val += 1
                accuracy_val = success_case_val/len(val_X)*100
                print('accuarcy in validation dataset is %.2f %%'%accuracy_val)
                
                accuracy = success_case/len(X)*100
                print('epoch %d:'%i,end='')
                print('accuracy in training is %.2f %%'%accuracy)
            if accuracy >= 100-1e-5:
                break
                
    def train_one_sample(self, x, y, learning_rate):
        output = self.forward(x)
        self.calc_gradient(y)
        self.update_weight(learning_rate)
        return output
    def forward(self, x):
        self.layers[0].forward(x)
        a = self.layers[0].y

        self.layers[1].forward(a)
        a = self.layers[1].y

        self.layers[2].forward(a)
        a = self.layers[2].y
        
        self.layers[3].forward(a)
        a = self.layers[3].y
        
        self.layers[4].forward(a)
        y = self.layers[4].y
        return y
    def calc_gradient(self, label):
        '''
        Your task 3
        - a) calculate the gradient of output layer
          (refer to the in Eq.2 of Section 3.3)
        
        - b) calculate the gradient of hidden layer
          (refer to the in Eq.3 of Section 3.4)
        '''
        
        # ****** Your code begin (task 3a) ******
        # out_delta = refrence calc_gradient in Section2_1layerNN.py
        out_delta = self.layers[4].activator.backward(self.layers[4].y) * (label - self.layers[4].y)
        # ****** Your code end (task 3a) ******
        
        self.layers[4].backward(out_delta)
        
        # ****** Your code begin (task 3b) ******
        # hid_delta = refrence Eq.3, the hid_delta is got from delta of output layer
        hid_delta = self.layers[4].delta
        # ****** Your code end (task 3b) ******
        
        self.layers[3].backward(hid_delta)
        hid_delta = self.layers[3].delta
 
        self.layers[2].backward(hid_delta)
        hid_delta = self.layers[2].delta
        
        self.layers[1].backward(hid_delta)
        hid_delta = self.layers[1].delta
        
        self.layers[0].backward(hid_delta)
        return self.layers[0].delta
        

    def update_weight(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

def data_processing():
        
    X,Y = load_data()
    train_X,train_Y = X[:1180],Y[:1180]
    val_X,val_Y = X[1180:1200],Y[1180:1200]
    train_X = train_X.reshape((*train_X.shape,1))
    train_Y = train_Y.reshape((*train_Y.shape,1))
    val_X = val_X.reshape((*val_X.shape,1))
    val_Y = val_Y.reshape((*val_Y.shape,1))
    #amount_of_c_in_train = train_Y.sum()
    #amount_of_c_in_val =val_Y.sum()
    #print('amount_of_c_in_train=%d'%amount_of_c_in_train)
    #print('amount_of_c_in_val=%d'%amount_of_c_in_val)
    return train_X,train_Y,val_X,val_Y

train_X,train_Y,val_X,val_Y = data_processing()
hidden_size = 45
model = Network(train_X.shape[1],hidden_size,train_Y.shape[1])
    
#training
epochs = 1000
learning_rate = 0.01
model.train(train_X,train_Y,epochs,learning_rate)
    
#validation
success_case = 0
for i in range(len(val_X)):
    #calssify the image is c if the output >=0.5
    if np.argmax(val_Y[i]) == np.argmax(model.forward(val_X[i])):
        success_case += 1
accuracy = success_case/len(val_X)*100
print('accuarcy in validation dataset is %.2f %%'%accuracy)
    
#save weights
np.set_printoptions(precision=24)
with open('save_weights.txt','w') as f:
    f.write(str(model))


    