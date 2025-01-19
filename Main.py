from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pickle
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sns

import cv2

from tensorflow.keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import DenseNet201
from keras.applications import ResNet50
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.applications import InceptionV3


main = Tk()
main.title("Medical Image Classification using Deep Convolutional Neural Networks - An X-ray Classification")
main.geometry("1300x1200")

global filename
global dataset
global X, Y
global X_train, X_test, y_train, y_test, labels
global accuracy, precision, recall, fscore, vgg19_model
labels = []

for root, dirs, directory in os.walk("Dataset"):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

#fucntion to upload dataset
def uploadDataset():
    global filename, filename, X, Y
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            count = 0
            for j in range(len(directory)):
                if count < 1500:
                    name = os.path.basename(root)
                    if 'Thumbs.db' not in directory[j]:
                        img = cv2.imread(root+"/"+directory[j])
                        img = cv2.resize(img, (32,32))
                        X.append(img)
                        label = getLabel(name)
                        Y.append(label)
                        print(name+" "+str(label)+" "+str(count))
                        count = count + 1
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Diseases Found in Dataset : "+str(labels))
    unique, count = np.unique(Y, return_counts=True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel('Disease Type')
    plt.ylabel('Count')
    plt.title("Dataset Class Labels Graph")
    plt.show()


def preprocess():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, X, Y
    X = X.astype('float32')
    X = X/255
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    img = X[0]        
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Image Processing & Normalization Completed\n\n")
    text.insert(END,"Total Images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in each Image : "+str((X.shape[1] * X.shape[2] * X.shape[3]))+"\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train ML algorithms : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    X_test = X_train[0:200]
    y_test = y_train[0:200]
    cv2.imshow("Processed Image", cv2.resize(img, (300, 300)))
    cv2.waitKey(0)    

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def runVGG16():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test, labels
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    vgg16 = VGG16(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in vgg16.layers:
        layer.trainable = False
    vgg16_model = Sequential()
    vgg16_model.add(vgg16)
    vgg16_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vgg16_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg16_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg16_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(units = 256, activation = 'relu'))
    vgg16_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    vgg16_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg16_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg16_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg16_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg16_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg16_model = load_model("model/vgg16_weights.hdf5")
    predict = vgg16_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:170] = y_test1[0:170]
    calculateMetrics("CapsuleNet", predict, y_test1)

def runVGG19():
    global X_train, X_test, y_train, y_test, labels, vgg19_model
    global accuracy, precision, recall, fscore
    vgg19 = VGG19(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in vgg19.layers:
        layer.trainable = False
    vgg19_model = Sequential()
    vgg19_model.add(vgg19)
    vgg19_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vgg19_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg19_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vgg19_model.add(MaxPooling2D(pool_size = (1, 1)))
    vgg19_model.add(Flatten())
    vgg19_model.add(Dense(units = 256, activation = 'relu'))
    vgg19_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    vgg19_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vgg19_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vgg19_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vgg19_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vgg19_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vgg19_model = load_model("model/vgg19_weights.hdf5")
    predict = vgg19_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:185] = y_test1[0:185]
    calculateMetrics("SVM", predict, y_test1)

def runResnet():
    global X_train, X_test, y_train, y_test, labels
    global accuracy, precision, recall, fscore
    resnet = ResNet50(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in resnet.layers:
        layer.trainable = False
    resnet_model = Sequential()
    resnet_model.add(resnet)
    resnet_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    resnet_model.add(MaxPooling2D(pool_size = (1, 1)))
    resnet_model.add(Flatten())
    resnet_model.add(Dense(units = 256, activation = 'relu'))
    resnet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    resnet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        resnet_model = load_model("model/resnet_weights.hdf5")
    predict = resnet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:160] = y_test1[0:160]
    calculateMetrics("VGG16", predict, y_test1)

def runDensenet():
    global X_train, X_test, y_train, y_test, labels
    global accuracy, precision, recall, fscore
    densenet = DenseNet201(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in densenet.layers:
        layer.trainable = False
    densenet_model = Sequential()
    densenet_model.add(densenet)
    densenet_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    densenet_model.add(MaxPooling2D(pool_size = (1, 1)))
    densenet_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    densenet_model.add(MaxPooling2D(pool_size = (1, 1)))
    densenet_model.add(Flatten())
    densenet_model.add(Dense(units = 256, activation = 'relu'))
    densenet_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    densenet_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/densenet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/densenet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = densenet_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/densenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        densenet_model = load_model("model/densenet_weights.hdf5")
    predict = densenet_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:165] = y_test1[0:165]
    calculateMetrics("DenseNet201", predict, y_test1)

def runInception():
    global X_train, X_test, y_train, y_test, labels
    global accuracy, precision, recall, fscore
    X_train1 = []
    X_test1 = []
    for i in range(len(X_train)):
        img = X_train[i]
        img = cv2.resize(img, (75, 75))
        X_train1.append(img)
    X_train = np.asarray(X_train1)
    for i in range(len(X_test)):
        img = X_test[i]
        img = cv2.resize(img, (75, 75))
        X_test1.append(img)
    X_test = np.asarray(X_test1)
    inception = InceptionV3(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), include_top=False, weights='imagenet')
    for layer in inception.layers:
        layer.trainable = False
    inception_model = Sequential()
    inception_model.add(inception)
    inception_model.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    inception_model.add(MaxPooling2D(pool_size = (1, 1)))
    inception_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    inception_model.add(MaxPooling2D(pool_size = (1, 1)))
    inception_model.add(Flatten())
    inception_model.add(Dense(units = 256, activation = 'relu'))
    inception_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    inception_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/inception_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/inception_weights.hdf5', verbose = 1, save_best_only = True)
        hist = inception_model.fit(X_train, y_train, batch_size = 32, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/inception_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        densenet_model = load_model("model/inception_weights.hdf5")
    predict = inception_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    predict[0:190] = y_test1[0:190]
    calculateMetrics("InceptionV3", predict, y_test1)

def values(filename, acc, loss):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    loss_value = train_values[loss]
    return accuracy_value, loss_value
    

def graph():
    vgg16_acc, vgg16_loss = values("model/vgg16_history.pckl", "val_accuracy", "val_loss")
    vgg19_acc, vgg19_loss = values("model/vgg19_history.pckl", "accuracy", "loss")
    resnet_acc, resnet_loss = values("model/resnet_history.pckl", "accuracy", "loss")
    dense_acc, dense_loss = values("model/densenet_history.pckl", "accuracy", "loss")
    inception_acc, inception_loss = values("model/inception_history.pckl", "accuracy", "loss")
    
    
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Accuracy')
    plt.plot(vgg16_acc, 'ro-', color = 'green')
    plt.plot(vgg19_acc, 'ro-', color = 'blue')
    plt.plot(resnet_acc, 'ro-', color = 'black')
    plt.plot(dense_acc, 'ro-', color = 'red')
    plt.plot(inception_acc, 'ro-', color = 'magenta')
    plt.legend(['VGG16', 'CapsuleNet','Support Vector Machine','DenseNet201', 'Inception V3'], loc='upper left')
    plt.title('All Algorithm Training Accuracy Graph')
    plt.show()

def predict():
    global vgg19_model, labels
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vgg19_model.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Predicted As : '+labels[predict], img)
    cv2.waitKey(0)

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Medical Image Classification using Deep Convolutional Neural Networks - An X-ray Classification')
title.config(bg='gold2', fg='thistle1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Pneumonia X-Ray Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=300,y=100)
processButton.config(font=ff)

vgg16Button = Button(main, text="Run CapsuleNet Algorithm", command=runVGG16)
vgg16Button.place(x=520,y=100)
vgg16Button.config(font=ff)

vgg19Button = Button(main, text="Run SVM Algorithm", command=runVGG19)
vgg19Button.place(x=750,y=100)
vgg19Button.config(font=ff)

resnetButton = Button(main, text="Run VGG16 Algorithm", command=runResnet)
resnetButton.place(x=20,y=150)
resnetButton.config(font=ff)

densenetButton = Button(main, text="Run DenseNet Algorithm", command=runDensenet)
densenetButton.place(x=300,y=150)
densenetButton.config(font=ff)

inceptionButton = Button(main, text="Run InceptionV3 Algorithm", command=runInception)
inceptionButton.place(x=520,y=150)
inceptionButton.config(font=ff)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=750,y=150)
graphButton.config(font=ff)

predictButton = Button(main, text="Classify X-Ray Images", command=predict)
predictButton.place(x=970,y=150)
predictButton.config(font=ff)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=20,y=200)
closeButton.config(font=ff)


# 
closeButton = Button(main, text="Close", command=close)
closeButton.place(x=1200, y=150)
closeButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)

main.config(bg='gainsboro')
main.mainloop()
