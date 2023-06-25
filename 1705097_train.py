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

np.random.seed(7)

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

    def backPropagation(self, loss_gradients, alpha):
        output_height=loss_gradients.shape[1]
        output_width=loss_gradients.shape[2]
        
        self.prev_output_image=np.pad(self.prev_output_image, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))

        del_X=np.zeros(self.prev_output_image.shape)
        del_K=np.zeros(self.weights.shape)
        del_B=np.zeros(self.biases.shape)
        padded_del_X=np.pad(del_X, ((0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)))

        for i in range(output_height):
            for j in range(output_width):
                v1=i*self.stride
                v2=j*self.stride

                for k in range(self.num_of_filters):
                    patch_image=self.prev_output_image[:, v1:v1+self.filter_dimension, v2:v2+self.filter_dimension, :]
                    reshaped_patch_image=patch_image.reshape(-1, self.filter_dimension*self.filter_dimension*self.num_of_channels)
                    reshaped_loss_gradients=loss_gradients[:, i, j, k].reshape(-1, 1)
                    reshaped_filters=self.weights[:, :, :, k].reshape(1, self.filter_dimension*self.filter_dimension*self.num_of_channels)

                    del_B[:, :, :, k]+=np.sum(loss_gradients[:, i, j, k])
                    del_K[:, :, :, k]+=np.dot(reshaped_loss_gradients.T, reshaped_patch_image)[0].reshape(self.filter_dimension, self.filter_dimension, self.num_of_channels)
                    padded_del_X[:, v1:v1+self.filter_dimension, v2:v2+self.filter_dimension, :]+=np.dot(reshaped_loss_gradients, reshaped_filters).reshape(-1, self.filter_dimension, self.filter_dimension, self.num_of_channels)

        if self.padding!=0:
            del_X[:, :, :, :]=padded_del_X[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:       
            del_X=padded_del_X

        self.weights-=(alpha*del_K)
        self.biases-=(alpha*del_B)  

        return del_X  

    def cleanParameters(self):
        self.num_of_channels=None
        self.input_dimension=None
        self.prev_output_image=None
#----------------------------------------------------ActivationLayer----------------------------------------------------------------------
class ActivationLayer:
    def __init__(self):
        self.prev_output_image=None

    def feedForward(self, image):
        self.prev_output_image=image
        output=np.copy(image)
        output[output<0]=0

        return output

    def backPropagation(self, loss_gradients, alpha):
        derivatives=np.copy(self.prev_output_image)
        derivatives[derivatives<0]=0
        derivatives[derivatives>0]=1
        del_X=loss_gradients*derivatives

        return del_X

    def cleanParameters(self):
        self.prev_output_image=None
#---------------------------------------------------------MaxPoolingLayer-----------------------------------------------------------------        
class MaxPoolingLayer:
    def __init__(self, filter_dimension, stride):
        self.filter_dimension=filter_dimension
        self.stride=stride
        self.prev_output_image=None

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

    def backPropagation(self, loss_gradients, alpha):
        output_height=loss_gradients.shape[1]
        output_width=loss_gradients.shape[2]
        num_of_channels=loss_gradients.shape[3]
        
        del_X=np.zeros(self.prev_output_image.shape)

        for i in range(output_height):
            for j in range(output_width):
                v1=i*self.stride
                v2=j*self.stride

                for k in range(num_of_channels):
                    patch_image=self.prev_output_image[:, v1:v1+self.filter_dimension, v2:v2+self.filter_dimension, k]
                    samples, r, c=patch_image.shape
                    reshaped_patch_image=patch_image.reshape(samples, r*c)
                    index_of_max=np.argmax(reshaped_patch_image, axis=1) + [t*r*c for t in range(samples)]
                    mask=np.zeros(patch_image.shape)
                    mask[np.unravel_index(index_of_max, patch_image.shape)]=1
                    del_X[:, v1:v1+self.filter_dimension, v2:v2+self.filter_dimension, k]+=mask*loss_gradients[:, i, j, k, np.newaxis, np.newaxis]
        
        return del_X

    def cleanParameters(self):
        self.prev_output_image=None
#-----------------------------------------------------------FlatteningLayer---------------------------------------------------------------
class FlatteningLayer:
    def __init__(self):
        self.prev_output_image=None

    def feedForward(self, image):
        self.prev_output_image=image
        output=image.flatten('C').reshape(image.shape[0], -1)           # 'C' means to flatten in row-major (C-style) order, -1 means unknown columns
        
        return output

    def backPropagation(self, loss_gradients, alpha):
        return np.reshape(loss_gradients, self.prev_output_image.shape)

    def cleanParameters(self):
        self.prev_output_image=None
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

    def backPropagation(self, loss_gradients, alpha):
        del_X=np.zeros(self.prev_output_image.shape)
        del_K=np.zeros(self.weights.shape)
        del_B=np.zeros(self.biases.shape)

        del_K=np.dot(loss_gradients.T, self.prev_output_image)          # https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/
        del_B=np.sum(loss_gradients.T, axis=1, keepdims=True)             # when summing along axis 1 (across rows), a 2D array is returned when keepdims = True
        del_X=np.dot(loss_gradients, self.weights)

        self.weights-=(alpha*del_K)
        self.biases-=(alpha*del_B)
        del_X=del_X.reshape(self.prev_output_image_shape)

        return del_X

    def cleanParameters(self):
        self.output_dimension=None
        self.prev_output_image=None
        self.prev_output_image_shape=None
#-----------------------------------------------------------SoftmaxLayer---------------------------------------------------------------
class SoftmaxLayer:
    def __init__(self) -> None:
        pass

    def feedForward(self, image):
        return np.exp(image.T)/np.sum(np.exp(image.T), axis=0)   

    def backPropagation(self, loss_gradients, alpha):
        return loss_gradients

    def cleanParameters(self)->None:
        pass 
#------------------------------------------------------ReadingData--------------------------------------------------------------------
def getLabels(path, name, dataset_portion):
    labels_dict = {}
    labels_temp = pd.read_csv(path)
    n=int((labels_temp.shape[0])*dataset_portion)
    filenames = labels_temp.iloc[:n,0]
    labels = labels_temp.iloc[:n,3]                          
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
            # titles = ['Original Image',"Binary Image",'Dilated Image']
            # images = [image, masked_image, dilated_image]
            # plt.figure(figsize=(13,5))
            # for i in range(3):
            #     plt.subplot(1,3,i+1)
            #     plt.imshow(images[i],'gray')
            #     plt.title(titles[i])
            #     plt.xticks([])
            #     plt.yticks([])
            # plt.tight_layout()
            # plt.show()
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

def forwardPasses(image, y_train, layers):
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

def backwardPasses(loss_gradients, layers, alpha):
    for layer in reversed(layers):
        loss_gradients=layer.backPropagation(loss_gradients, alpha)

def train(image, y_train, layers, alpha):
    num_of_samples=image.shape[0]
    y_predicted, training_loss, training_accuracy, training_macro_f1=forwardPasses(image, y_train, layers)

    # One Hot Encoding
    y_true_onehot = np.zeros((y_predicted.shape))
    y_true_onehot[np.arange(y_train.shape[0]), y_train]=1
    # for i in range(y_true_onehot.shape[1]):
    #     y_true_onehot[np.arange(y_train.shape[0]), i] = 1
    
    loss_gradients=np.ones((y_predicted.shape)) 
    loss_gradients=(y_predicted-y_true_onehot)/num_of_samples 
    loss_gradients=backwardPasses(loss_gradients, layers, alpha)

    return training_loss, training_accuracy, training_macro_f1

def runTrainingAndValidation(X_train, y_train, layers, epoch, alpha, num_batch_samples, num_of_batches, X_validation, y_validation):
    # max_f1_score=-9999999  
    for e in range(epoch):  
        total_loss=0
        total_accuracy=0
        print("-------------Epoch number: ",e+1,"---------------")

        for i in range(num_of_batches):
            n_samples = y_train.shape[0] - i * num_batch_samples if (i + 1) * num_batch_samples > y_train.shape[0] else num_batch_samples
            training_loss, training_accuracy, training_macro_f1=train( X_train[i*num_batch_samples : i*num_batch_samples+n_samples], y_train[i*num_batch_samples : i*num_batch_samples+n_samples], layers, alpha)
            if (i+1)%100==0:
                print("Upto batch ",i+1, " training completed")
            total_loss+=training_loss
            total_accuracy+=training_accuracy
   
        total_loss=(total_loss/X_train.shape[0])
        total_accuracy=(total_accuracy/X_train.shape[0])*100
        print('Training Loss: {:.3f}%'.format(total_loss))
        print('Training accuracy: {:.3f}%'.format(total_accuracy))
        y_predicted, validation_loss, validation_accuracy, validation_macro_f1=forwardPasses(X_validation, y_validation, layers)
        validation_loss=validation_loss/X_validation.shape[0]
        validation_accuracy=(validation_accuracy/X_validation.shape[0])*100
        print('Validation loss: {:.3f}%'.format(validation_loss))
        print('Validation accuracy: {:.3f}%'.format(validation_accuracy))
        print('Validation macro-f1: {:.3f}'.format(validation_macro_f1))
        if (e+1)==epoch:
            # predicted_label=np.argmax(y_predicted, axis=1) 
            # cm=confusion_matrix(y_validation, predicted_label)
            # print(cm)
            # fx=plot(y_validation, predicted_label) 
            # plt.savefig('cm_model_3_alpha_0.1.jpg')
            # plt.show()
        # if validation_macro_f1 > max_f1_score:
        #     max_f1_score=validation_macro_f1
            for layer in layers:
                layer.cleanParameters()

            with open('model_3_alpha_0.05_pickle','wb') as f:
                pickle.dump(layers, f)

#-------------------------------------------------------Main----------------------------------------------------------------------
IMG_SIZE=32

# Getting training labels from 3 directories
labels_a = getLabels("NumtaDB_with_aug/training-a.csv", "labels_a", 0.15)
labels_b = getLabels("NumtaDB_with_aug/training-b.csv", "labels_b", 0.30)
labels_c = getLabels("NumtaDB_with_aug/training-c.csv", "labels_c", 0.07)

# Getting training images and their respective ground truth values from 3 directories
path_a = "NumtaDB_with_aug/training-a"
path_b = "NumtaDB_with_aug/training-b"
path_c = "NumtaDB_with_aug/training-c"

train_a, labels_a = getData(path_a, IMG_SIZE, labels_a, "train_a")
train_b, labels_b = getData(path_b, IMG_SIZE, labels_b, "train_b")
train_c, labels_c = getData(path_c, IMG_SIZE, labels_c, "train_c")

# Merging all individual subsets of training images and labels into one training set.
X_trainValid = train_a+train_b+train_c                                   # X_trainValid contains images reading grayscale mode and then resized
y_trainValid = labels_a+labels_b+labels_c                                # y_trainValid contains digits

X_trainValid = np.asarray(X_trainValid)
X_trainValid = normalize(X_trainValid)
y_trainValid = np.asarray(y_trainValid)

X_trainValid=np.reshape(X_trainValid,(X_trainValid.shape[0],X_trainValid.shape[1],X_trainValid.shape[2],1))
print("Shape of X_trainValid: ", X_trainValid.shape)
print("Shape of y_trainValid: ", y_trainValid.shape)

X_train=X_trainValid
y_train=y_trainValid
X_validation=X_trainValid
y_validation=y_trainValid
print("Dataset prepared!")

#---------------------------Architecture----------------------------
layers=[]
dimension=X_train.shape[1]
channel_count=X_train.shape[-1]

input_file=open("input3.txt","r")
for l in input_file:
    str=l.split(" ")
    layer_name=str[0].strip()
    conv_input=dimension*dimension*channel_count
    if layer_name=="conv":
        Conv=ConvolutionLayer(int(str[1]), int(str[2]), int(str[3]), int(str[4]), channel_count,conv_input)
        channel_count=int(str[1])
        dimension=floor((dimension-int(str[2])+(2*int(str[4])))/int(str[3]))+1
        layers.append(Conv)
    
    elif layer_name=="relu": 
        Relu=ActivationLayer()
        layers.append(Relu)
            
    elif layer_name=="pool":
        Pool=MaxPoolingLayer(int(str[1]), int(str[2]))
        dimension=floor((dimension-int(str[2]))/int(str[2]))+1
        layers.append(Pool)
    
    elif layer_name=="flatten":
        flatten=FlatteningLayer()
        layers.append(flatten)

    elif layer_name=="fc":
        FC=FullyConnectedLayer(int(str[1]))
        layers.append(FC)

    elif layer_name=="softmax":
        Softmax=SoftmaxLayer()
        layers.append(Softmax)

input_file.close()

#---------------------------Running----------------------------
alpha=0.05
epoch=7
num_batch_samples=32
num_of_batches=floor(y_train.shape[0]/num_batch_samples)
runTrainingAndValidation(X_train, y_train, layers, epoch, alpha, num_batch_samples, num_of_batches, X_validation, y_validation)