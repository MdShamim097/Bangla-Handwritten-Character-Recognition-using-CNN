import pandas as pd
import numpy as np
from math import floor
import cv2
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import seaborn as sns 
import pickle
import csv
import sys

np.random.seed(7)
filename_list=[]

#------------------------------------------------------ConvolutionLayer--------------------------------------------------------------------
class ConvolutionLayer:
    def __init__(self, num_of_filters, filter_dimension, stride, padding, num_of_channels, input_dimension):
        self.num_of_filters=num_of_filters
        self.filter_dimension=filter_dimension
        self.stride=stride
        self.padding=padding
        self.num_of_channels=num_of_channels
        self.input_dimension=input_dimension
        self.prev_output_image=None
        self.weights=np.random.randn(self.filter_dimension,self.filter_dimension,self.num_of_channels,self.num_of_filters)*(np.sqrt(2/self.input_dimension))
        self.biases=np.zeros((1, 1, 1, self.num_of_filters))

    def feedForward(self, image):
        self.prev_output_image=image
        num_of_samples=image.shape[0]
        input_height=image.shape[1]
        input_width=image.shape[2]

        output_height=floor((input_height-self.filter_dimension+2*self.padding)/self.stride)+1
        output_width=floor((input_width-self.filter_dimension+2*self.padding)/self.stride)+1
        output=np.zeros((num_of_samples, output_height, output_width, self.num_of_filters))

        padded_image=np.pad(image, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))

        for i in range(output_height):
            for j in range(output_width):
                v1=i*self.stride
                v2=j*self.stride
                patch_image=padded_image[:, v1:v1+self.filter_dimension, v2:v2+self.filter_dimension, :]

                for k in range(self.num_of_filters):
                    v3=np.multiply(patch_image, self.weights[:, :, :, k])
                    output[:, i, j, k]=np.sum(v3, axis=(1,2,3)) + self.biases[:, :, :, k]
        
        return output

#----------------------------------------------------ActivationLayer----------------------------------------------------------------------
class ActivationLayer:
    def __init__(self):
        self.prev_output_image=None
        self.weights=None
        self.biases=None

    def feedForward(self, image):
        self.prev_output_image=image
        output=np.copy(image)
        output[output<0]=0

        return output

#---------------------------------------------------------MaxPoolingLayer-----------------------------------------------------------------        
class MaxPoolingLayer:
    def __init__(self, filter_dimension, stride):
        self.filter_dimension=filter_dimension
        self.stride=stride
        self.prev_output_image=None
        self.weights=None
        self.biases=None

    def feedForward(self, image):
        self.prev_output_image=image
        num_of_samples=image.shape[0]
        input_height=image.shape[1]             # if batch of images are sent then shape 1, 2 will be height, width
        input_width=image.shape[2]
        num_of_channels=image.shape[3]          # if not batch

        output_height=floor((input_height-self.filter_dimension)/self.stride)+1
        output_width=floor((input_width-self.filter_dimension)/self.stride)+1
        output=np.zeros((num_of_samples, output_height, output_width, num_of_channels))

        for i in range(output_height):
            for j in range(output_width):
                v1=i*self.stride
                v2=j*self.stride
                patch_image=image[:, v1:v1+self.filter_dimension, v2:v2+self.filter_dimension, :]
                output[:, i, j, :]=np.max(patch_image, axis=(1,2))
        
        return output

#-----------------------------------------------------------FlatteningLayer---------------------------------------------------------------
class FlatteningLayer:
    def __init__(self):
        self.prev_output_image=None
        self.weights=None
        self.biases=None

    def feedForward(self, image):
        self.prev_output_image=image
        output=image.flatten('C').reshape(image.shape[0], -1)           # 'C' means to flatten in row-major (C-style) order, -1 means unknown columns
        
        return output

#-----------------------------------------------------FullyConnectedLayer---------------------------------------------------------------------
class FullyConnectedLayer:
    def __init__(self, output_dimension):
        self.output_dimension=output_dimension
        self.prev_output_image=None
        self.prev_output_image_shape=None   
        self.weights=None
        self.biases=None      

    def feedForward(self, image):
        self.prev_output_image_shape=image.shape
        self.prev_output_image=image 
        
        if self.weights is None:
            self.weights=np.random.randn(self.output_dimension, image.shape[1])*(np.sqrt(2/image.shape[1]))            # https://cs231n.github.io/neural-networks-2/#init
        if self.biases is None:
            self.biases=np.zeros((self.output_dimension, 1))

        output=np.dot(self.weights, image.T) + self.biases
        output=output.T

        return output

#-----------------------------------------------------------SoftmaxLayer---------------------------------------------------------------
class SoftmaxLayer:
    def __init__(self):
        self.weights=None
        self.biases=None

    def feedForward(self, image):
        return np.exp(image.T)/np.sum(np.exp(image.T), axis=0)   

#------------------------------------------------------ReadingData--------------------------------------------------------------------
def getLabels(path, name, dataset_portion):
    labels_dict = {}
    labels_temp = pd.read_csv(path)
    n=int((labels_temp.shape[0])*dataset_portion)
    filenames = labels_temp.iloc[:n, 1]
    labels = labels_temp.iloc[:n, 0]   

    for i in range(len(filenames)):
        filename_list.append(filenames[i])

    for item in  range(len(labels)):
        labels_dict[filenames[item]] = labels[item]
    print(f"{name}: Loading complete!")
    return labels_dict

def getData(images_path, IMG_SIZE, labels_dict, name):
    train_set = []
    y = []
    
    for fname in os.listdir(images_path):
        if fname in labels_dict:
            y.append(labels_dict[fname])
            image = cv2.imread(os.path.join(images_path,fname), 0)
            retVal, masked_image=cv2.threshold(image, 205, 255, cv2.THRESH_BINARY_INV)          # cv2.THRESH_BINARY_INV keeps foreground white & background black
            kernel = np.ones((3, 3), np.uint8)
            dilated_image=cv2.dilate(masked_image, kernel, iterations=3)
            img_array = cv2.resize(dilated_image, (IMG_SIZE, IMG_SIZE))
            train_set.append(img_array)
        else:
            pass

    print(f"{name}: Loading complete!")
    return train_set, y    

def normalize(x):
    x = x/255.0
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x

#------------------------------------------------------Training--------------------------------------------------------------------
def plot(y_true, y_pred):
    labels=unique_labels(y_true)
    column=[f'Predicted {label}' for label in labels]
    indices=[f'Actual {label}' for label in labels]
    table=pd.DataFrame(confusion_matrix(y_true, y_pred), columns=column, index=indices)
    return sns.heatmap(table, annot=True, fmt='d', cmap='viridis')

def forwardPasses(image, layers, y_train):
    img=image
    for layer in layers:
        img=layer.feedForward(img)  

    y_predicted=img.T                               # (num_batch_samples, FC layer output dimension) 

    y_true = np.zeros((y_predicted.shape))
    y_true[np.arange(y_train.shape[0]), y_train]=1
    # for i in range(y_true.shape[1]):
    #     y_true[np.arange(y_train.shape[0]), i] = 1
 
    loss=(np.sum(-1 * np.sum(y_true * np.log(y_predicted), axis=0)))
    
    predicted_label=np.argmax(y_predicted, axis=1)  
    accuracy=np.count_nonzero(np.equal(predicted_label, y_train))
    # accuracy=accuracy_score(y_train, predicted_label)
    macro_f1=f1_score(y_train, predicted_label, average='macro')            # 'macro' : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.  

    return y_predicted, loss, accuracy, macro_f1

def runTestingSet(X_test, y_test, layers):
    y_predicted, test_loss, test_accuracy, test_macro_f1=forwardPasses(X_test, layers, y_test)
    return y_predicted, test_loss, test_accuracy, test_macro_f1
#-------------------------------------------------------Main----------------------------------------------------------------------
IMG_SIZE=32

# Getting training labels from 3 directories
labels_d = getLabels("NumtaDB_with_aug/test-b2.csv", "labels_d", 1)

# Getting training images and their respective ground truth values from 3 directories
path_d = "NumtaDB_with_aug/test-b2"

train_d, labels_d = getData(path_d, IMG_SIZE, labels_d, "train_d")

X_test = train_d                               
y_test = labels_d 

X_test = np.asarray(X_test)
X_test = normalize(X_test)
y_test = np.asarray(y_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
print("Shape of X_test: ", X_test.shape)
print("Shape of y_test: ", y_test.shape)
print("Dataset prepared!")
#---------------------------Architecture----------------------------
with open('1705097_model.pickle','rb') as f:
    layers=pickle.load(f)

#---------------------------Running----------------------------
y_predicted, test_loss, test_accuracy, test_macro_f1=runTestingSet(X_test, y_test, layers)
test_loss=test_loss/X_test.shape[0]
test_accuracy=(test_accuracy/X_test.shape[0])*100
print('Test loss: {:.3f}%'.format(test_loss))
print('Test accuracy: {:.3f}%'.format(test_accuracy))
print('Test macro-f1: {:.3f}'.format(test_macro_f1))

predicted_label=np.argmax(y_predicted, axis=1)

TestResultsDir=sys.argv[1]
print("Path of directory: ", TestResultsDir, end = " ")
if not os.path.exists('TestResultsDir/'):
    os.makedirs('TestResultsDir/')

rows=[]
for i in range(X_test.shape[0]):
    t=[]
    t.append(filename_list[i])
    t.append(predicted_label[i])
    rows.append(t)

fields = ['Filename', 'Digit'] 

with open('TestResultsDir/independent_test_results.csv', 'w') as csv_file:
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(fields) 
    csv_writer.writerows(rows)

fx=plot(y_test, predicted_label) 
plt.savefig('TestResultsDir/cm_independent_test.jpg')
plt.show()