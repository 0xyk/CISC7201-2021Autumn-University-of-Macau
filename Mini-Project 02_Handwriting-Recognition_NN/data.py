
#data.py
# load the images
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.core.fromnumeric import reshape 
    
def convert_image(image):
    image = image.sum(axis=2)//3
    return image

    
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
    X = X.reshape(X.shape[0],-1)
    Y = np.array(list(map(one_hot,Y)))
    return X,Y


    
if __name__ == '__main__':
    #plot the first figure as a sample
    letters = [chr(x+ord('A')) for x in range(26)]
    X,Y = load_images()
    print('The letter is %s'%letters[Y[0]])
    plt.imshow(X[0])
    plt.show()
    

