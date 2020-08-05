# -*- coding: utf-8 -*-


import os
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
import imutils
import matplotlib
import seaborn as sns
import matplotlib.animation as animation
from mpl_toolkits.basemap import Basemap
from ast import literal_eval
import pandas as pd

#directory where all the images are stored
dir = os.path.join(os.getcwd(), "drive/My Drive/data")

def loadData(dir):

    images = []
    category = []
    file_dir = []

    for img in os.listdir(dir):

        if img.endswith(".png"):
        #images.append(cv2.imread(os.path.join(dir, img)))
            file_dir.append(os.path.join(dir, img))

            if img.startswith('NL'):      #assigning label to file according to          
                category.append(0)             # their type

            elif img.startswith('ca'):    
                category.append(1)

            elif img.startswith('Gl'):  
                category.append(2)

            elif img.startswith('Re'):          
                category.append(3)

    file_dir = np.array(file_dir)
    category = np.array(category, dtype = 'int32')  

    return shuffle(file_dir, category,random_state=862)

file_dir, category = loadData(dir)

print(len(file_dir))

#mount drive to use dataset uploaded
from google.colab import drive
drive.mount('/content/drive')

def get_label(encoded_val):
      
      labels = {0:'normal', 1:'cataract', 2:'glaucoma', 3:'retina'}
      return labels[encoded_val]

def data_augment(file_dir, labels):

    img_set = []
    lbl_set = []

    for i in range(len(file_dir)):

        if i%100==0:
            print(i)
        label = labels[i]
        img = get_object(file_dir[i])

        img_b = img + 0.07*img
        img_d = img - 0.07*img

        flip_v_b = cv2.flip(img_b,0)
        flip_h_b = cv2.flip(img_b,1)

        flip_v_d = cv2.flip(img_d,0)
        flip_h_d = cv2.flip(img_d,1)

        flip_v = cv2.flip(img,0) #vertical flip
        flip_h = cv2.flip(img,1) #lr flip

        img_set.append(img)
        # img_set.append(img_b)
        # img_set.append(img_d)
        img_set.append(flip_v)
        img_set.append(flip_h)
        img_set.append(flip_v_b)
        img_set.append(flip_h_b)
        img_set.append(flip_v_d)
        img_set.append(flip_h_d)

        # lbl_set.append(label)
        # lbl_set.append(label)
        lbl_set.append(label)
        lbl_set.append(label)
        lbl_set.append(label)
        lbl_set.append(label)
        lbl_set.append(label)
        lbl_set.append(label)
        lbl_set.append(label)
    print('Done')
    return np.array(img_set),np.array(lbl_set)

def get_object(filename):

    image = cv2.imread(filename)
    im_size = 256
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # convert 2 grayscale

    retval, threshold = cv2.threshold(gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours

    # ensure at least some circles were found
    if contours:

        contours = sorted(contours, key=cv2.contourArea, reverse=True) 

        #find the bounding rect
        x,y,w,h = cv2.boundingRect(contours[0])                  
        img = image[y:y+h,x:x+w]# crop image

        resize=cv2.resize(img,(im_size,im_size)) # resize to im_size X im_size size
        
        return resize

    return cv2.resize(image, (im_size,im_size))

X, y = data_augment(file_dir, category)

"""CNN model"""

import keras
y_encoded = keras.utils.to_categorical(y, 4)
X = X.astype('float32')
X /= 255

import seaborn as sns
data = pd.DataFrame(
    data=(get_label(y[i]) for i in range(0, len(y))),    # values
    columns=['eye condition']
) 

input = sns.countplot(
    data=data,
    x = 'eye condition',
    order = ['retina', 'glaucoma', 'cataract', 'normal']
).set_title('Input Data')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=5)

Y_train = np.argmax(y_train,axis = 1)
data = pd.DataFrame(
    data=(get_label(Y_train[i]) for i in range(0, len(Y_train))),    # values
    columns=['eye condition']
) 

training = sns.countplot(
    data=data,
    x = 'eye condition',
    order = ['retina', 'glaucoma', 'cataract', 'normal']
).set_title('Training Data')

plt.savefig('Training Data')

Y_test = np.argmax(y_test,axis = 1)
data = pd.DataFrame(
    data=(get_label(Y_test[i]) for i in range(0, len(Y_test))),    # values
    columns=['eye condition']
) 

training = sns.countplot(
    data=data,
    x = 'eye condition',
    order = ['retina', 'glaucoma', 'cataract', 'normal']
).set_title('Test Data')
plt.savefig('Test Data')

Y_val = np.argmax(y_val,axis = 1)
data = pd.DataFrame(
    data=(get_label(Y_val[i]) for i in range(0, len(Y_val))),    # values
    columns=['eye condition']
) 

val = sns.countplot(
    data=data,
    x = 'eye condition',
    order = ['retina', 'glaucoma', 'cataract', 'normal']
).set_title('Validation Data')

plt.savefig('Validation Data')

from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import keras

"""To visualize results"""

def visualize_results(history):

    # Plot the accuracy and loss curves
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    
    plt.title('Training and validation loss')
    
    plt.legend()
    plt.show()
    plt.savefig(f'{history}.png')

"""Creating model using Keras"""

trained_models = []

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

"""Model 2"""

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(4, activation='softmax')(x)

# this is the model we will train
model_2 = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
model_2.summary()
# compile the model (should be done *after* setting layers to non-trainable)
model_2.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# model.summary()
# train the model on the new data for a few epochs
trained_model_1 = model_2.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=30, #number of iterations
    validation_data=(X_val, y_val),
    shuffle=True
)

_, accuracy = model_2.evaluate(X_test, y_test)
print(f'The model accuracy is {accuracy}')
visualize_results(trained_model_1)

from sklearn.metrics import confusion_matrix
prediction = model_2.predict(X_test)
Y_pred_classes = np.argmax(prediction,axis = 1)

# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

print(confusion_mtx)

plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d")
plt.savefig("trained_model_1_cm.png")

lt.rcParams['animation.ffmpeg_path'] = "C:\FFmpeg\\bin\\ffmpeg.exe"


def get_data(table, rownum, title):
    data = pd.DataFrame(table.loc[rownum])
    data.columns = {title}
    return data


def augment(xold, yold, numsteps):
    xnew = []
    ynew = []
    for i in range(len(xold)-1):
        difX = xold[i+1]-xold[i]
        stepsX = difX/numsteps
        difY = yold[i+1]-yold[i]
        stepsY = difY/numsteps
        for s in range(numsteps):
            xnew = np.append(xnew, xold[i]+s*stepsX)
            ynew = np.append(ynew, yold[i]+s*stepsY)
    return xnew, ynew


def smoothListGaussian(listin, strippedXs=False, degree=5):
    window = degree*2-1
    weight = np.array([1.0]*window)
    weightGauss = []
    for i in range(window):
        i = i-degree+1
        frac = i/float(window)
        gauss = 1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)
    weight = np.array(weightGauss)*weight
    smoothed = [0.0]*(len(listin)-window)
    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(listin[i:i+window])*weight)/sum(weight)
    return smoothed


csv = pd.read_csv('./dataone.csv')


x = list(csv["Year"].astype('category').value_counts().index)
x.sort()
csv.sort_values(by=['Year'], inplace=True)
y = list(csv["Year"].value_counts().sort_index())
XN, YN = augment(x, y, 10)
XN = smoothListGaussian(XN)
YN = smoothListGaussian(YN)
# augmented =
print(y)

print(x)

title = 'Eye Diseases'

df = pd.DataFrame(YN, XN)
#XN,YN = augment(x,y,10)
#augmented = pd.DataFrame(YN,XN)
df.columns = {title}

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)


# fig = plt.figure(figsize=(10, 6))
# plt.xlim(2005, 2016)
# plt.ylim(np.min(df)[0], np.max(df)[0])
# plt.xlabel('Year', fontsize=20)
# plt.ylabel(title, fontsize=20)
# plt.title('Eye Diseases per Year', fontsize=20)

#used to create the GIF for presentation
def animate(i):
    data = df.iloc[:int(i+1)]  # select data range
    p = sns.lineplot(x=data.index, y=data[title], data=data, color="r")
    p.tick_params(labelsize=17)
    plt.setp(p.lines, linewidth=7)


sns.set(rc={'axes.facecolor': 'lightgrey', 'figure.facecolor': 'lightgrey',
            'figure.edgecolor': 'black', 'axes.grid': False})
# ani = matplotlib.animation.FuncAnimation(fig, animate, frames=150, repeat=True)
# ani.save('test.mp4', writer=writer)
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6,
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

for loc in csv["GeoLocation"].unique():
    weight = csv["GeoLocation"].value_counts()[loc]//250
    lat, long = literal_eval(loc)
    x, y = m(long, lat)
    plt.plot(x, y, 'ok', markersize=weight)

plt.show()
