from tqdm import tqdm
import numpy as np
from utils import cross_entropy_loss, softmax, predict, one_hot, save_weight, save_history
from time import time
from sklearn.metrics import accuracy_score
class Parameter:
    def __init__(self, learning_rate=0.01, iteration=1000, lambda1=0, lambda2=0, momentum=0, alpha=0.3, beta=0.8):
        self.learning_rate=learning_rate
        self.iteration=iteration
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.momentum=momentum
        self.alpha=alpha
        self.beta=beta

    def to_dict(self):
        return{
        "learning_rate": self.learning_rate,
        "iteration": self.iteration,
        "lambda1": self.lambda1,
        "lambda2": self.lambda2,
        "momentum": self.momentum,
        "alpha": self.alpha,
        "beta": self.beta

        }        


class BaseModel:
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "iteration", experiment_name = "my_experiment"):
        assert mode in ["time", "iteration"]
        self.num_classes = num_classes
        self.learning_rate=learning_rate
        self.num_iterations=num_iterations
        self.do_one_hot=do_one_hot
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.momentum=momentum
        self.alpha=alpha
        self.beta=beta
        self.mode = mode
        self.experiment_name = experiment_name
        self.early_stop = early_stop

        self.training_info = {"experiment_name": experiment_name,
                     "model_name": "softmax_regression_gd",
                     "mode": mode,
                     "early_stop": early_stop,
                      "history": []
                     }


    def train(self, X_train, y_train, X_val, y_val):
       
        

        if self.do_one_hot:
            y_one_hot_train = one_hot(y_train, self.num_classes)
            y_one_hot_val = one_hot(y_val, self.num_classes)
        else:
            y_one_hot_train = y_train
            y_train = np.where(y_one_hot_train == 1)[1]

            y_one_hot_val = y_val
            y_val = np.where(y_one_hot_val == 1)[1]

        

        start_iter = time()
        best_accuracy = 0
        early_stop_counter = 0
        W = None
        b = None
        for i in tqdm(range(1, self.num_iterations+1)):        
            W, b = self.train_iter(X_train, y_one_hot_train, i, W, b)
    
            end_iter = time()
            
            params = Parameter(learning_rate=self.learning_rate,
                                iteration=i, lambda1=self.lambda1, lambda2=self.lambda2, momentum=self.momentum, alpha=self.alpha, beta=self.beta)
            

            if self.mode == "iteration":
                
                y_pred = predict(X_train, W, b)
                loss = cross_entropy_loss(y_one_hot_train, y_pred) + self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2)
                if i % 10 == 0:
                    print(f"Iteration {i}, Loss: {loss}")

                params.__setattr__("loss", loss)

                y_val_pred = predict(X_val, W, b)
                val_loss = cross_entropy_loss(y_one_hot_val, y_val_pred) + self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2)
                params.__setattr__("val_loss", val_loss)

                Y_val_pred = np.argmax(predict(X_val, W, b), axis=1)

                val_acc = accuracy_score(y_val, Y_val_pred)
                params.__setattr__("val_acc", val_acc)
                if val_acc >= best_accuracy:
                    best_accuracy = val_acc
                    early_stop_counter = 0
                    self.training_info["best_accuracy"] = best_accuracy
                else:
                    early_stop_counter +=1
                if i % 10 == 0:
                    print(f"Iteration {i}, Val accuracy: {val_acc}")

            if early_stop_counter == self.early_stop:
                print("Early stopping condition reached, stopping training")
                break
            if self.mode == "time":
                params.__setattr__("weight", [W, b])
                params.__setattr__("elapsed_time", end_iter - start_iter)
                
                self.training_info["history"].append(params.to_dict())
            
            self.training_info["history"].append(params)
        return W, b, self.training_info

    def train_iter(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError
