import numpy as np
from utils import cross_entropy_loss, softmax, predict, one_hot, save_weight, save_history
from base_model import BaseModel

class SoftmaxRegression(BaseModel):
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "iteration", do_backtrack = False, experiment_name = "softmax_reg_gd"):
        super().__init__(num_classes, learning_rate, num_iterations, do_one_hot, lambda1, lambda2, momentum, alpha, beta, early_stop, mode, experiment_name)
        self.do_backtrack = do_backtrack

    # momentum: thay đổi momentum
    # alpha=0.3, beta=0.8
    def __init_params__(self, X):
        self.m, self.n = X.shape
        W = np.random.randn(self.n, self.num_classes) * 0.01
        b = np.zeros((1, self.num_classes))
        self.v_w = np.zeros_like(W)
        self.v_b = np.zeros_like(b)

        return W, b
    
    def train_iter(self, X_train, y_train, t, W = None, b = None):
        
        if t == 1:
            W, b = self.__init_params__(X_train)
        
        v_w = self.v_w
        v_b = self.v_b

        logits = np.dot(X_train, W) + b
        y_pred = softmax(logits)

        # Compute loss
        # loss = cross_entropy_loss(y, y_pred) + self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2)

        # Backward pass (compute gradients) with L1 and L2 computation
        dw = (
            (1 / self.m) * np.dot(X_train.T, (y_pred - y_train))
            + self.lambda1 * np.sign(W)
            + self.lambda2 * W
        )
        db = (1 / self.m) * np.sum(y_pred - y_train, axis=0, keepdims=True)
        
        # Backtracking line search
        if self.do_backtrack:
            while True:
                v_w_prev = v_w
                v_b_prev = v_b

                v_w = self.momentum * v_w - self.learning_rate * dw
                v_b = self.momentum * v_b - self.learning_rate * db

                weights_temp = W - self.momentum * v_w_prev + (1 + self.momentum) * v_w
                bias_temp = b - self.momentum * v_b_prev + (1 + self.momentum) * v_b

                linear_output_temp = np.dot(X_train, weights_temp) + bias_temp
                y_pred_temp = softmax(linear_output_temp)
                loss_temp = cross_entropy_loss(y_train, y_pred_temp)

                if loss_temp <= cross_entropy_loss(y_train, y_pred) - self.alpha * self.learning_rate * (
                    np.linalg.norm(dw) ** 2 + np.linalg.norm(db) ** 2
                ):
                    break
                else:
                    self.learning_rate *= self.beta

            W = weights_temp
            b = bias_temp
        else:
            v_w_prev = v_w
            v_b_prev = v_b

            v_w = self.momentum * v_w - self.learning_rate * dw
            v_b = self.momentum * v_b - self.learning_rate * db

            W = W - self.momentum * v_w_prev + (1 + self.momentum) * v_w
            b = b - self.momentum * v_b_prev + (1 + self.momentum) * v_b
        self.v_w = v_w
        self.v_b = v_b
        return W, b


def adam_optimizer(W, b, dw, db, mW, mb, vW, vb, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda1 = 0, lambda2 = 0):
    # Update biased first moment estimate
    mW = beta1 * mW + (1 - beta1) * dw
    mb = beta1 * mb + (1 - beta1) * db

    # Update biased second raw moment estimate
    vW = beta2 * vW + (1 - beta2) * (dw ** 2)
    vb = beta2 * vb + (1 - beta2) * (db ** 2)

    # Compute bias-corrected first moment estimate
    mW_hat = mW / (1 - beta1 ** t)
    mb_hat = mb / (1 - beta1 ** t)

    # Compute bias-corrected second raw moment estimate
    vW_hat = vW / (1 - beta2 ** t)
    vb_hat = vb / (1 - beta2 ** t)

    # Apply L1 and L2 regularization
    W_reg = lambda1 * np.sign(W) + lambda2 * W

    # Update parameters
    W -= learning_rate * (mW_hat / (np.sqrt(vW_hat) + epsilon) + W_reg)
    b -= learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)

    return W, b, mW, mb, vW, vb




class SoftmaxAdam(BaseModel):
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "iteration", experiment_name = "softmax_adam", beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(num_classes, learning_rate, num_iterations, do_one_hot, lambda1, lambda2, momentum, alpha, beta, early_stop, mode, experiment_name)
        
        self.beta1=beta1 
        self.beta2=beta2
        self.epsilon=epsilon
        self.lambda1 = lambda1, 
        self.lambda2= lambda2
        self.training_info["model_name"] = "softmax_adam"
        
    def __init_params__(self, X):
        self.m, self.n = X.shape
        W = np.random.randn(self.n, self.num_classes) * 0.01
        b = np.zeros((1, self.num_classes))
        self.mW, self.mb = np.zeros_like(W), np.zeros_like(b)
        self.vW, self.vb = np.zeros_like(W), np.zeros_like(b)
        return W, b
    def train_iter(self, X_train, y_train, t, W = None, b = None):
        if t == 1:
            W, b = self.__init_params__(X_train)
        
        mW, mb = self.mW, self.mb
        vW, vb = self.vW, self.vb
        logits = np.dot(X_train, W) + b
        y_pred = softmax(logits)

        # Compute loss
        # loss = cross_entropy_loss(y, y_pred)

        # Backward pass (compute gradients)
        dw = (1 / self.m) * np.dot(X_train.T, (y_pred - y_train))
        db = (1 / self.m) * np.sum(y_pred - y_train, axis=0, keepdims=True)

        # Update weights and biases using Adam
        W, b, mW, mb, vW, vb = adam_optimizer(W, b, dw, db, mW, mb, vW, vb, t, self.learning_rate, self.beta1, self.beta2, self.epsilon)
        self.mW, self.mb, self.vW, self.vb = mW, mb, vW, vb    
        return W, b

if  __name__ == "__main__":

    import os
    import pickle as pkl

    # vers = ["", "_3_turns", "_5_turns", "_7_turns", "_8_turns", "_9_turns"]
    vers = [ "_5_turns"]
    os.makedirs("history", exist_ok = True)

    val_acc = {}
    for ver in vers:        
        X_train = np.load(f"split_data/X{ver}_train.npy")
        Y_train = np.load(f"split_data/Y{ver}_train.npy")
        X_val = np.load(f"split_data/X{ver}_val.npy")
        Y_val = np.load(f"split_data/Y{ver}_val.npy")
        
        print(f"training with ver {ver}")
        # model = SoftmaxAdam(num_classes= Y_val.shape[1], early_stop= 100, learning_rate=0.01, do_one_hot= False, experiment_name=f"exp{ver}")
        model = SoftmaxRegression(num_classes= Y_val.shape[1], early_stop= 100, learning_rate=0.01, do_one_hot= False, experiment_name=f"exp{ver}")
        model.train(X_train, Y_train, X_val, Y_val)
        
        val_acc[f"exp{ver}"] = model.training_info["best_accuracy"]
        
        with open(f"history/exp{ver}.pkl", "wb") as f:
            pkl.dump(model.training_info, f)
    
    sorted_dict= dict(sorted(val_acc.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)