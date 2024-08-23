from mini_batch_algo import *
from full_batch_algo import *
import pickle as pkl

learning_rates = [0.005, 0.01, 0.05, 0.1]
l1_l2 = [(0, 0), (0.001, 0), (0.01, 0), (0.1, 0), 
                (0, 0.001), (0.01, 0), (0, 0.1)]
batch_sizes = [1]
ver = "_5_turns"
X_train = np.load(f"split_data/X{ver}_train.npy")
Y_train = np.load(f"split_data/Y{ver}_train.npy")
X_val = np.load(f"split_data/X{ver}_val.npy")
Y_val = np.load(f"split_data/Y{ver}_val.npy")
val_acc_mini = {}
for learning_rate in learning_rates:
    for l1, l2 in l1_l2:
        for batch_size in batch_sizes:
            l1 = l1 / X_train.shape[1]
            l2 = l2 / X_train.shape[1]
            
            expriment_name = f"softmax_regression_sgd_lr_{learning_rate}_l1_{l1}_l2_{l2}_batch_size_{batch_size}"
            print(expriment_name)
            model = SoftmaxRegressionSGD(num_classes= Y_val.shape[1], num_iterations = 1000, early_stop = 50, learning_rate=learning_rate, batch_size = batch_size, lambda1 = l1, lambda2 = l2, do_one_hot= False, experiment_name=expriment_name)
            model.train(X_train, Y_train, X_val, Y_val)
            val_acc_mini[expriment_name] = model.training_info["best_accuracy"]
            print(val_acc_mini)
            with open(f"history/{expriment_name}.pkl", "wb") as f:
                pkl.dump(model.training_info, f)