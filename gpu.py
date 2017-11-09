#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

## gpu demo################
# 使用高斯噪聲
import time
from keras.layers.noise import GaussianDropout
# # input image dimensions
img_rows, img_cols = 256,256
path1 = './image'    #path of folder of images    
path2 = './input'  #path of folder to save images 
listing = os.listdir(path1) 
num_samples=size(listing)
print (num_samples)

for file in listing:
    im = Image.open(path1 + '/' + file)   
    img = im.resize((img_rows,img_cols))
    color = img.convert('RGBA') #L為灰階 RGBA為彩色
                #need to do some more processing here           
    color.save(path2 +'/' +  file, "JPEG")

imlist = os.listdir(path2)

im1 = array(Image.open(path2 + '/'+ imlist[0])) # open one image to get size

m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')

label=np.ones((num_samples,),dtype = int)
label[0:114]=0  # 二條城
label[114:1233]=1  # 三十三間堂
label[1233:3789]=2  # 千本鳥居
label[3789:5639]=3  # 平等院鳳凰堂 
label[5639:5928]=4  # 京都御所
label[5928:7857]=5  # 京都塔
label[7857:9809]=6  # 金閣寺
label[9809:10294]=7  # 清水寺
label[10294:11392]=8  # 渡月橋
data,Label = shuffle(immatrix,label, random_state=4)
train_data = [data,Label]

#batch_size to train
# batch_size = 32
# number of output classes 有幾種lagel
nb_classes = 9
# number of epochs to train
nb_epoch = 30


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(x, y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

x_train = x_train.reshape(x_train.shape[0], 128, 128,3).astype('float32')

x_test = x_test.reshape(x_test.shape[0], 128, 128,3).astype('float32')

# 標準化
x_train_normalize = x_train / 255
x_test_normalize = x_test / 255
y_train_OneHot = np_utils.to_categorical(y_train, nb_classes)
y_test_OneHot = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(2,2),
                 padding='same',
                 input_shape=(256, 256,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=64,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=128,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=256,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=512,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Conv2D(filters=1024,
                 kernel_size=(2,2),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(GaussianDropout(0.3))

model.add(Flatten())

model.add(Dense(1000,activation='relu'))
model.add(GaussianDropout(0.3))
model.add(Dense(500,activation='relu'))
model.add(GaussianDropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(GaussianDropout(0.3))
model.add(Dense(9,activation='softmax'))
print(model.summary())


#############################################################################################################
print(model.summary())
import tensorflow as tf
print(tf.test.is_gpu_available())
with tf.device('/gpu:0'):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'] ) #optimizer='adadelta'
    a=time.clock()
    train_history = model.fit(x_train_normalize, y_train_OneHot,
                              validation_data=(x_test_normalize, y_test_OneHot),
                              epochs=50,
                              batch_size=100,verbose=1) 
    print('train_history')	    
