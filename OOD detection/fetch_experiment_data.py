import os
import cv2
import numpy as np
from sklearn.datasets import fetch_openml

def load_test_images(folder_path):
    X_list = []
    y_list = []

    for filename in os.listdir(folder_path):
        # 1. Extract the class label
        name_without_ext = os.path.splitext(filename)[0]
        if '_' in name_without_ext:
            label = name_without_ext.split('_')[0]
    
        # 2. Read the image in grayscale (28x28)
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            X_list.append(img)
            y_list.append(label)
        else:
            print(f"Warning: Could not read image {filename}")

    # Convert lists to NumPy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y

def load_mnist_zeros(start: int = 0, stop: int = 90):
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    # Only get zeros from start to stop (default is the first 90)
    mask = y == "0"
    X = X[mask][start:stop]
    y = y[mask][start:stop]
    return X, y

def get_group_A():
    FOLDER_PATH_GROUP_A = "Test data\drawings class A"

    X_zero, y_zero = load_mnist_zeros()
    X_self_drawn, y_self_drawn = load_test_images(FOLDER_PATH_GROUP_A)

    X_self_drawn_flat = X_self_drawn.reshape(len(X_self_drawn), -1)  # Becomes (180, 784) -> same shape as the zeros

    X = np.concatenate((X_self_drawn_flat, X_zero), axis=0)  # New shape: (270, 784)
    y = np.concatenate((y_self_drawn, y_zero), axis=0)       # New shape: (270,)

    return X, y

def get_group_B():
    FOLDER_PATH_GROUP_A = "Test data\drawings class B"

    class_mapper = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9"
    }

    X_self_drawn, y_self_drawn = load_test_images(FOLDER_PATH_GROUP_A)
    X_self_drawn_flat = X_self_drawn.reshape(len(X_self_drawn), -1)

    # Covert the written classes back to the same classes of the model:
    for i in range(len(y_self_drawn)):
       written_class = y_self_drawn[i]
       y_self_drawn[i] = class_mapper[written_class]

    return X_self_drawn_flat, y_self_drawn

if __name__ == "__main__":
    pass
    X, y = get_group_B()