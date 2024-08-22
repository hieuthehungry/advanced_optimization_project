import os
import pickle as pkl
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm
# vers = ["", "_3_turns", "_5_turns", "_7_turns", "_8_turns", "_9_turns"]
vers = [ "_5_turns"]
os.makedirs("history", exist_ok = True)

gbm_params = {
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimator": 100, # int, 100->1000 (step 100)
        "verbosity": -1 }


val_acc = {}
ver = vers[0]

X_train = np.load(f"split_data/X{ver}_train.npy")
Y_train = np.load(f"split_data/Y{ver}_train.npy")
X_val = np.load(f"split_data/X{ver}_val.npy")
Y_val = np.load(f"split_data/Y{ver}_val.npy")
# for n_estimator in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
#     copy_params = gbm_params
#     copy_params["n_estimator"] = n_estimator
    
    
#     print(f"training with ver {ver} and {n_estimator} estimators")
#     # model = SoftmaxRegressionMiniAdam(num_classes= Y_val.shape[1], early_stop= 100, learning_rate=0.01, do_one_hot= False, alpha = 0, beta = 0.9, experiment_name=f"exp{ver}")
#     model = LGBMClassifier(**copy_params).fit(X_train, np.argmax(Y_train, axis = 1))
#     Y_pred = model.predict(X_val)
#     acc = accuracy_score(np.argmax(Y_val, axis = 1), Y_pred)
#     val_acc[n_estimator] = acc
    
#     # with open(f"history/exp_light_gbm{ver}.pkl", "wb") as f:
#     #     pkl.dump(model.training_info, f)
    
# sorted_dict= dict(sorted(val_acc.items(), key=lambda item: item[1], reverse=True))
# print(sorted_dict)
# with open(f"history/exp_light_gbm{ver}.json", "w") as f:
#     json.dump(val_acc, f)

# best_n_estimator = list(sorted_dict.keys())[0]
# print(f"Best n_estimator {best_n_estimator}")
feature_report = {}
for i in tqdm(range(X_train.shape[1])):
    copy_params = gbm_params
    copy_params["n_estimator"] = 100
    X_train_temp =X_train[:, i].reshape(-1, 1)
    Y_train = Y_train
    X_val_temp = X_val[:, i].reshape(-1, 1)
    Y_val = Y_val
    # model = SoftmaxRegressionMiniAdam(num_classes= Y_val.shape[1], early_stop= 100, learning_rate=0.01, do_one_hot= False, alpha = 0, beta = 0.9, experiment_name=f"exp{ver}")
    model = LGBMClassifier(**copy_params).fit(X_train_temp, np.argmax(Y_train, axis = 1))
    Y_pred = model.predict(X_val_temp)
    acc = accuracy_score(np.argmax(Y_val, axis = 1), Y_pred)

    feature_report[i]= acc

print(min(feature_report.values()))
print(max(feature_report.values()))
with open(f"history/exp_light_gbm{ver}_feature_acc.json", "w") as f:
    json.dump(feature_report, f)