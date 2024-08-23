from mini_batch_algo import *
from full_batch_algo import *
from newton_with_reg import *
import pickle as pkl

learning_rates = [0.01, 0.1, 1]
l2s = [0, 0.001, 0.01, 0.1]
batch_sizes = [1]
ver = "_5_turns"
X_train = np.load(f"split_data/X{ver}_train.npy")
Y_train = np.load(f"split_data/Y{ver}_train.npy")
X_val = np.load(f"split_data/X{ver}_val.npy")
Y_val = np.load(f"split_data/Y{ver}_val.npy")
val_acc_mini = {}
for learning_rate in learning_rates:
    for l2 in l2s:
            if learning_rate in [0.01] and l2 in [0.0, 0.001]:
                print(f"This {learning_rate} and {l2} have been run, passed to next setting") 
                continue
            
            l2 = l2/X_train.shape[1]
            model = SoftmaxRegressionNewton(num_classes= Y_val.shape[1], num_iterations = 25, early_stop = 5, learning_rate=learning_rate, lambda2 = l2, do_one_hot= False)
            expriment_name = f"{model.training_info['model_name']}_lr_{learning_rate}_l2_{l2}"
            print(expriment_name)
            
            model.train(X_train, Y_train, X_val, Y_val)
            val_acc_mini[expriment_name] = model.training_info["best_accuracy"]
            with open(f"history/{expriment_name}.pkl", "wb") as f:
                pkl.dump(model.training_info, f)