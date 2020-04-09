import tensorflow as tf
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.backend import set_session
import cv2, os, gc
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class CNNClassifier:
    """
    Main Classifier class which can be used to classify the FingerVein Dataset.
    The default CNN Feature extractor used is ResNet50. However, any other architecture can be used.
    """

    def __init__(self, shape=None, enableGPUTraining=True):

        """
        Constructor of the class: Initialzies the object variables and makes sure that tensorflow 
        does not take up all the memory at once. Custom Image Dims can be used. Default is 340, 320
        """

        if shape is None:
            self.resize = False
            self.imshape = (240, 320, 3)

        else:
            self.resize = True
            self.imshape = shape


        self.model = None
        self.x = list()
        self.y = list()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preds = None

        if enableGPUTraining:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            set_session(sess)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    def init_model(self, num_classes):
        """
        Initialize the model architecture. The base architecture can be changed here.
        NOTE: Input image needs to be RGB (3 channels) while using any pre-defined architecture,
        Like ResNet50. For single channel grayscale images, custom architecture is required.
        """
        
        self.num_classes = num_classes

        # Load the ImageNet weights for the ResNet50 feature extractor
        base = Xception(include_top=False, weights='imagenet', input_shape=self.imshape)
        for layer in base.layers[:5]:
            layer.trainable = False

        x = base.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(250, activation='relu')(x)
        x = keras.layers.Dropout(0.25)(x)
        preds = keras.layers.Dense(num_classes, activation='softmax', name='fcFinal')(x)

        self.model = keras.models.Model(input=base.input, output=preds)

        # Make sure all the layers are trainable.
        for layer in self.model.layers:
            layer.trainable = True
        
        return self.model

    def load_data(self, root_path, test_set_percent=0.20, limitedMemory=True):
        """
        Load the data into memory. Also splits the data into training and testing/validation sets. 
        incase the machine has limited memory, set the argument limitedMemory=True in order to use the
        python garbage collector and clear the memory. 
        Test/Validation set size fraction can be specified form the arguments. Default is 0.2 (20%) of total data.
        """
        folder_names = sorted(os.listdir(root_path))

        # Read and pre-process the data from the path
        for folder in folder_names:
            left = glob(os.path.join(root_path, folder, 'left', '*'))
            right = glob(os.path.join(root_path, folder, 'right', '*'))
            imPaths = left + right
            imPaths = [path for path in imPaths if '.bmp' in path]

            for imPath in imPaths:
                image = cv2.imread(imPath)
                if self.resize:
                    image = cv2.resize(image, self.imshape[:2], cv2.INTER_AREA)
                self.x.append(image)
                self.y.append(int(folder))

        # Convert to Numpy array and convert to categorical data
        self.y = keras.utils.to_categorical(self.y)
        self.x = np.asarray(self.x)

        # Train-Test split and shuffle data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_set_percent, stratify=self.y)

        # delete unnecessary variables for limited memeory operations
        if limitedMemory:
            del self.x, self.y
            gc.collect()

        # Normalize the Images
        self.X_train = (self.X_train / 255.).astype(np.float16)
        self.X_test = (self.X_test / 255.).astype(np.float16)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self, lr=0.001, epoch=100, batch_size=64, weights_path='./weights/'):

        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

        callback_checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_path, 'weights.h5'), \
            monitor='val_accuracy', verbose=0, save_weights_only=True, save_best_only=True)
        callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, verbose=1)
        callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_lr=1e-7, patience=2, verbose=1)
        callbacks = [callback_checkpoint, callback_early_stopping, callback_reduce_lr]

        opt = keras.optimizers.Adam(lr=lr)
        self.model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x=self.X_train, y=self.y_train, nb_epoch=epoch, batch_size=batch_size, validation_data=(self.X_test, self.y_test), callbacks=callbacks)

        return self.model, history

    def predict(self, X=None, y=None, weights_path='./weights/'):
        """
        Function to predict on the default dataset or a dataset that is passed as an argument.
        It loads the saved weights and attempts to predict on the given data.
        It also computes the loss on the given dataset and returns the same.
        """

        if X is None and y is None:
            X = self.X_test
            y = self.y_test

        try:
            if len(X.shape) < 4:
                X = np.expand_dims(X, axis=0)
        except Exception as e:
            print("Error:", e)
            print("Please Load the data before trying to predict")
            exit(0)

        try:
            self.model.load_weights(os.path.join(weights_path, 'weights.h5'))
        except Exception as e:
            print("Error:", e)
            print("Please run init_model() before trying to predict")
            exit(0)

        self.preds = self.model.predict(X, workers=0, use_multiprocessing=True)
        self.preds = np.argmax(self.preds, axis=1)
        loss = mean_squared_error(np.argmax(y, axis=1), self.preds)
        return self.preds, loss


    def generate_confusion_matrix(self, show_graph=True):
        """
        Compute and generate the Confusion Matrix.
        NOTE: Please run this after running model.predict(). Without the preds, 
        confusion matrix won't be generated
        """
        try:
            matrix = confusion_matrix(np.argmax(self.y_test, axis=1), self.preds)
        except Exception as e:
            print("Error:", e)
            print("Please run model.predict before trying to compute the confusion matrix")
            exit(0)
        
        if show_graph:
            df_cm = pd.DataFrame(matrix)
            sns.heatmap(df_cm, annot=True)
            plt.show()
        return matrix

    
def display_curve(train_vals, val_vals, datatype="Accuracy"):
    """
    Display the Accuracy or Loss curves (training and testing) for trained network
    """
    plt.plot(train_vals,'r',linewidth=3.0, label='Training '+datatype)
    plt.plot(val_vals,'b',linewidth=3.0, label='Testing '+datatype)
    plt.legend(fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel(datatype, fontsize=16)
    plt.title(datatype+' Curves', fontsize=16)
    plt.show()

def train(root_path):
    # Declare the object
    network = CNNClassifier()

    # Initialize the Model and generate the architecture
    network.init_model(num_classes=107)

    # Load and preprocess the dataset
    network.load_data(root_path)

    # Train the model
    model, history = network.train(lr=0.0001, batch_size=16)

    # Save the model history as a pickle file (to access it later)
    with open('model_history.pkl', 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test(root_path):
    # Declare the object
    network = CNNClassifier()

    # Initialize the Model and generate the architecture
    network.init_model(num_classes=107)

    # Load and preprocess the dataset
    network.load_data(root_path)

    # Load the history and display training graphs
    with open('model_history.pkl', 'rb') as handle:
        history = pickle.load(handle)

    # Display Accuracy Curve
    display_curve(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy')
    
    # Display Loss Curve
    display_curve(history.history['loss'], history.history['val_loss'], 'Loss')

    # Make the predictions on the default test set. (pass custom test set if required)
    preds, loss = network.predict()

    # Generate the Confusion Matrix (Prediction is mandatry to do this step. Do not skip the previous step)
    network.generate_confusion_matrix(show_graph=True)


if __name__ == '__main__':
    root_path = 'data'

    # train(root_path)
    test(root_path)