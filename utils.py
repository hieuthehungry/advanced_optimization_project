import numpy as np
from tqdm import tqdm
import pickle as pkl

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability improvement by subtracting max
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]

    log_likelihood =  -np.sum(y_true * np.log(y_pred))
    loss = np.sum(log_likelihood) / m
    return loss

def one_hot(y, num_classes):
    m = y.shape[0]
    one_hot_matrix = np.zeros((m, num_classes))
    one_hot_matrix[np.arange(m), y] = 1
    return one_hot_matrix

def predict(X, W, b):
    logits = np.dot(X, W) + b
    return softmax(logits)


def save_weight(file_path, W, b):
    with open(file_path, "wb") as f:
        pkl.dump([W, b], f)
    
def load_weight(file_path):
    with open(file_path, "rb") as f:
        W, b = pkl.load(f)
    return W, b

def save_history(file_path, history):

   with open(file_path, "wb") as f:
        pkl.dump(history, f)
