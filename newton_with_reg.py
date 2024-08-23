import numpy as np
from utils import cross_entropy_loss, softmax, predict, one_hot, save_weight, save_history
from base_model import BaseModel

class SoftmaxRegressionNewton(BaseModel):
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=20, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "iteration", experiment_name = "softmax_reg_newton"):
        super().__init__(num_classes, learning_rate, num_iterations, do_one_hot, lambda1, lambda2, momentum, alpha, beta, early_stop, mode, experiment_name)
    
        self.training_info["model_name"] = "softmax_newton"
    
    def __init_params__(self, X):
        self.m, self.n = X.shape
        W = np.random.randn(self.n, self.num_classes) * 0.01
        b = np.zeros((1, self.num_classes))
        self.v_w = np.zeros_like(W)
        self.v_b = np.zeros_like(b)

        return W, b
    
    def train_iter(self, X_train, y_train, t, W = None, b = None):
        
        if t == 1:
            W, b  = self.__init_params__(X_train)
        
        v_w = self.v_w
        v_b = self.v_b

        logits = np.dot(X_train, W) + b
        y_pred = softmax(logits)

        # Compute loss
        # loss = cross_entropy_loss(y, y_pred) + self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2)

        # Backward pass (compute gradients) with L1 and L2 computation
        W, b = newton_optimizer(X_train, y_train, W, b, self.learning_rate, self.do_one_hot, self.lambda1, self.lambda2)

        return W, b
# X = np.load("X.npy")
# y = np.load("Y.npy")
# num_classes = y.shape[1]


# # Train the model
# W, b, loss_history = softmax_minibatch_adam(X, y, num_classes, batch_size = 64, do_one_hot = False, learning_rate=0.001, num_iterations=2000)

# # Make predictions
# y_pred = predict(X, W, b)
# predicted_classes = np.argmax(y_pred, axis=1)

# print("Predicted classes:", predicted_classes)

def newton_optimizer(X, y, W, b, learning_rate=1.0, do_one_hot = False, lambda1=0.0, lambda2=0.0):
    m, n = X.shape
    num_classes = W.shape[1]
    # print(num_classes)
    if do_one_hot:
      y_one_hot = one_hot(y, num_classes)
    else:
      y_one_hot = y
      y = np.where(y_one_hot==1)[1]

    logits = np.dot(X, W) + b
    y_pred = softmax(logits)

    # Compute gradient
    grad_W = (1 / m) * np.dot(X.T, (y_pred - y_one_hot)) + lambda1 * np.sign(W) + lambda2 * W
    grad_b = (1 / m) * np.sum(y_pred - y_one_hot, axis=0, keepdims=True)

    # Compute Hessian
    H_W = np.zeros((n, n, num_classes))
    H_b = np.zeros((1, 1, num_classes))

    # print(H_W.shape)
    # print(H_b.shape)
    for k in range(num_classes):
        S_k = np.diag(y_pred[:, k] * (1 - y_pred[:, k]))
        # print((np.dot(X.T, np.dot(S_k, X)) / m + lambda2 * np.eye(n)).shape)
        H_W[:, :, k] = np.dot(X.T, np.dot(S_k, X)) / m + lambda2 * np.eye(n)

        # Compute Hessian for bias
        # print((np.sum(S_k, axis=0) / m).shape)

        H_b[:, :, k] = (np.sum(S_k) / m).reshape(1, 1)

    # Update weights and biases using Newton-Raphson method
    for k in range(num_classes):
        H_W_inv = np.linalg.pinv(H_W[:, :, k])  # Pseudo-inverse in case Hessian is singular
        W[:, k] -= learning_rate * np.dot(H_W_inv, grad_W[:, k])

        # print((learning_rate * grad_b[:, k] / H_b[:, :, k]).squeeze().shape)
        # print(b[:, k].shape)
        b[:, k] -= (learning_rate * grad_b[:, k] / H_b[:, :, k]).squeeze()

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
        model = SoftmaxRegressionNewton(num_classes= Y_val.shape[1], early_stop= 5, learning_rate=1, do_one_hot= False)
        model.train(X_train, Y_train, X_val, Y_val)
        
        val_acc[f"exp{model.training_info['model_name']}{ver}"] = model.training_info["best_accuracy"]
        
        with open(f"history/exp{ver}.pkl", "wb") as f:
            pkl.dump(model.training_info, f)
    
    sorted_dict= dict(sorted(val_acc.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
