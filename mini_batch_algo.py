import numpy as np
from utils import cross_entropy_loss, softmax, predict, one_hot, save_weight, save_history
from base_model import BaseModel

class SoftmaxRegressionSGD(BaseModel):
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "time", experiment_name = "softmax_reg_gd", batch_size = 256, do_shuffle = True):
        self.super().__init__(num_classes, learning_rate, num_iterations, do_one_hot, lambda1, lambda2, momentum, alpha, beta, early_stop, mode, experiment_name)
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle



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

        

        # Compute loss
        if self.do_shuffle:
          indices = np.random.permutation(self.m)
          X_shuffled = X_train[indices]
          y_shuffled = y_train[indices]
        else:
          X_shuffled = X_train
          y_shuffled = y_train
        for i in range(0, self.m, self.batch_size):
            X_batch = X_shuffled[i:i + self.batch_size]
            y_batch = y_shuffled[i:i + self.batch_size]
            logits = np.dot(X_batch, W) + b
            y_pred = softmax(logits)
            
            # Backward pass (compute gradients) with L1 and L2 computation
            dw = (
                (1 / self.m) * np.dot(X_train.T, (y_pred - y_train))
                + self.lambda1 * np.sign(W)
                + self.lambda2 * W
            )
            db = (1 / self.m) * np.sum(y_pred - y_train, axis=0, keepdims=True)

            # Backtracking line search
            while True:
                v_w_prev = v_w
                v_b_prev = v_b

                v_w = self.momentum * v_w - self.learning_rate * dw
                v_b = self.momentum * v_b - self.learning_rate * db

                weights_temp = W - self.momentum * v_w_prev + (1 + self.momentum) * v_w
                bias_temp = b - self.momentum * v_b_prev + (1 + self.momentum) * v_b

                linear_output_temp = np.dot(X_batch, weights_temp) + bias_temp
                y_pred_temp = softmax(linear_output_temp)
                loss_temp = cross_entropy_loss(y_train, y_pred_temp) + self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2)

                if loss_temp <= cross_entropy_loss(y_batch, y_pred)+ self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2) - self.alpha * self.learning_rate * (
                    np.linalg.norm(dw) ** 2 + np.linalg.norm(db) ** 2
                ):
                    break
                else:
                    self.learning_rate *= self.beta

                W = weights_temp
                b = bias_temp
        self.v_w = v_w
        self.v_b = v_b
        return W, b
    


class SoftmaxAdam(BaseModel):
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "time", experiment_name = "softmax_adam", beta1=0.9, beta2=0.999, epsilon=1e-8,  lambda1 = 0, lambda2= 0, batch_size = 256, do_shuffle = True):
        self.super().__init__(num_classes, learning_rate, num_iterations, do_one_hot, lambda1, lambda2, momentum, alpha, beta, early_stop, mode, experiment_name)
        self.beta1=beta1, 
        self.beta2=beta2
        self.epsilon=epsilon
        self.lambda1 = lambda1, 
        self.lambda2= lambda2
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle

    def __init_params__(self, X):
        self.m, self.n = X.shape
        W = np.random.randn(self.n, self.num_classes) * 0.01
        b = np.zeros((1, self.num_classes))
        self.mW, self.mb = np.zeros_like(self.W), np.zeros_like(self.b)
        self.vW, self.vb = np.zeros_like(self.W), np.zeros_like(self.b)
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
        dw = (1 / self.m) * np.dot(X.T, (y_pred - y_train))
        db = (1 / self.m) * np.sum(y_pred - y_train, axis=0, keepdims=True)

        # Update weights and biases using Adam
        W, b, mW, mb, vW, vb = adam_optimizer(W, b, dw, db, mW, mb, vW, vb, t, self.learning_rate, self.beta1, self.beta2, self.epsilon)
        self.mW, self.mb, self.vW, self.vb = mW, mb, vW, vb    
        return W, b
    



class SoftmaxRegressionMiniAdam(BaseModel):
    def __init__(self, num_classes, learning_rate=0.01, num_iterations=1000, do_one_hot=False, lambda1=0, lambda2=0,
    momentum=0, alpha=0.3, beta=0.8, early_stop = 5, mode = "time", experiment_name = "softmax_reg_gd", batch_size = 256, do_shuffle = True):
        self.super().__init__(num_classes, learning_rate, num_iterations, do_one_hot, lambda1, lambda2, momentum, alpha, beta, early_stop, mode, experiment_name)
        self.batch_size = batch_size
        self.do_shuffle = do_shuffle



    def __init_params__(self, X):
        self.m, self.n = X.shape
        W = np.random.randn(self.n, self.num_classes) * 0.01
        b = np.zeros((1, self.num_classes))
        self.vW = np.zeros_like(W)
        self.vb = np.zeros_like(b)

        return W, b
    
    def train_iter(self, X_train, y_train, t, W = None, b = None):
        
        if t == 1:
            W, b = self.__init_params__(X_train)
        
        vW = self.vW
        vb = self.vb

        

        # Compute loss
        if self.do_shuffle:
          indices = np.random.permutation(self.m)
          X_shuffled = X_train[indices]
          y_shuffled = y_train[indices]
        else:
          X_shuffled = X_train
          y_shuffled = y_train
        for i in range(0, self.m, self.batch_size):
            X_batch = X_shuffled[i:i + self.batch_size]
            y_batch = y_shuffled[i:i + self.batch_size]
            logits = np.dot(X_batch, W) + b
            y_pred = softmax(logits)
            
            # Backward pass (compute gradients) with L1 and L2 computation
            dw = (
                (1 / self.m) * np.dot(X_train.T, (y_pred - y_train))
                + self.lambda1 * np.sign(W)
                + self.lambda2 * W
            )
            db = (1 / self.m) * np.sum(y_pred - y_train, axis=0, keepdims=True)

            # Backtracking line search
            while True:
                v_w = self.momentum * v_w - self.learning_rate * dw
                v_b = self.momentum * v_b - self.learning_rate * db

                dw = (1 / self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (1 / self.batch_size) * np.sum(y_pred - y_batch, axis=0, keepdims=True)

                # Update weights and biases using Adam with regularization
                weights_temp, bias_temp, mW, mb, vW, vb = adam_optimizer(W, b, dw, db, mW, mb, vW, vb, t, self.learning_rate, self.beta1, self.beta2, self.epsilon, self.lambda1, self.lambda2)

                linear_output_temp = np.dot(X_batch, weights_temp) + bias_temp
                y_pred_temp = softmax(linear_output_temp)
                loss_temp = cross_entropy_loss(y_train, y_pred_temp) + self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2)

                if loss_temp <= cross_entropy_loss(y_batch, y_pred)+ self.lambda1 * np.sum(np.abs(W)) +  self.lambda2 * np.sum(W**2) - self.alpha * self.learning_rate * (
                    np.linalg.norm(dw) ** 2 + np.linalg.norm(db) ** 2
                ):
                    break
                else:
                    self.learning_rate *= self.beta

                W = weights_temp
                b = bias_temp
        self.v_w = v_w
        self.v_b = v_b
        return W, b
    





def adam_optimizer(W, b, dw, db, mW, mb, vW, vb, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda1=0.0, lambda2=0.0):
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


def softmax_minibatch_adam(X, y, num_classes, batch_size, do_one_hot = True, learning_rate=0.001, num_iterations=1000, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda1=0.0, lambda2=0.0):
    m, n = X.shape
    W = np.random.randn(n, num_classes) * 0.01
    b = np.zeros((1, num_classes))

    if do_one_hot:
      y_one_hot = one_hot(y, num_classes)
    else:
      y_one_hot = y
      y = np.where(y_one_hot==1)[1]

    # Initialize moment estimates
    mW, mb = np.zeros_like(W), np.zeros_like(b)
    vW, vb = np.zeros_like(W), np.zeros_like(b)

    loss_history = []
    for t in tqdm(range(1, num_iterations + 1)):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y_one_hot[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            # Forward pass
            logits = np.dot(X_batch, W) + b
            y_pred = softmax(logits)

            # Compute loss
            loss = cross_entropy_loss(y_batch.argmax(axis=1), y_pred)

            # Backward pass (compute gradients)
            dw = (1 / batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1 / batch_size) * np.sum(y_pred - y_batch, axis=0, keepdims=True)

            # Update weights and biases using Adam with regularization
            W, b, mW, mb, vW, vb = adam_optimizer(W, b, dw, db, mW, mb, vW, vb, t, learning_rate, beta1, beta2, epsilon, lambda1, lambda2)

        if t % 100 == 0:
            print(f"Iteration {t}, Loss: {loss}")

        logits = np.dot(X, W) + b
        y_pred = softmax(logits)
        loss = cross_entropy_loss(y, y_pred)
        loss_history.append(loss)

    return W, b, loss_history