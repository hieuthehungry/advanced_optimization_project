from sklearn.model_selection import train_test_split
import os 
import numpy as np

index = np.load("vector_data/index.npy")
os.makedirs("split_data", exist_ok = True)

X_file = "X"
Y_file = "Y"
vers = ["", "_3_turns", "_5_turns", "_7_turns", "_8_turns", "_9_turns"]

index_of_index = list(range(len(index)))
train_index, valid_index = train_test_split(index_of_index, random_state = 42)


train, valid = index[train_index], index[valid_index]

np.save(f"split_data/train.npy", train)
np.save(f"split_data/valid.npy", valid)
for ver in vers:
    X = np.load(f"vector_data/X{ver}.npy")
    Y = np.load(f"vector_data/Y{ver}.npy")
    
    X_train, X_val = X[train_index], X[valid_index]
    Y_train, Y_val = Y[train_index], Y[valid_index]

    np.save(f"split_data/X{ver}_train.npy", X_train)
    np.save(f"split_data/Y{ver}_train.npy", Y_train)
    np.save(f"split_data/X{ver}_val.npy", X_val)
    np.save(f"split_data/Y{ver}_val.npy", Y_val)