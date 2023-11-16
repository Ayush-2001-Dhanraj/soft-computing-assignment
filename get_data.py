import numpy as np
import cv2
import os

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path, num_images=10):
    """
    returns samples and lables specified by path and dataset params

    loop through each label and append image to X and label to y
    """
    labels = os.listdir(os.path.join(path, dataset))

    X = []
    y = []
    
    for label in labels:
        image_counter = 0
        for file in os.listdir(os.path.join(path, dataset, label)):
            if image_counter < num_images:
                print(image_counter)
                image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
                X.append(image)
                y.append(label)
                image_counter += 1

    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
    """
    returns train X, y and test X and y
    """
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    return X, y, X_test, y_test
