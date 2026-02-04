import tensorflow as tf
import keras

def load_data():
    # Load CIFAR-10
    # images with 32×32 pixels and 10 classes
    (X_train, y_train) , (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize
    # pixel values range : 0-255
    # convert pixel values in to a range 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return X_train, y_train, X_test, y_test