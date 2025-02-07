import numpy as np
class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, constant_step_size=True, true_weights=None):
        self.lr = learning_rate
        self.is_constant = constant_step_size
        self.weights = None
        self.err = None
        self.true_weights = true_weights

    # Step-size update for implementation with decaying step-size
    def update_lr(self, iter):
        if not self.is_constant:
            updated_lr = self.lr/iter
        else:
            updated_lr = self.lr
        return updated_lr

    # Run the sgd algorithm over all the data samples
    def run_sgd(self, X, y, cost_funct):
        n_features, n_iters = X.shape #righe per colonne, le colonne sono il n di dati quindi n iterazioni
        self.weights = np.zeros((n_features, n_iters))
        self.err = np.zeros(n_iters)
        for i in range(1,n_iters): #parto da 1 perchè i pesi della prima iterazione (i-1) sono 0, quindi inizio ad aggiornare come la formula
            if cost_funct == 'mse': #mi computa il gradiente dando i parametri
                grad = compute_gradient_linreg(self.weights[:,i-1], X[:,i], y[i]) #gli dà una x singola per aggiornare i pesi
            elif cost_funct == 'logloss':
                grad = compute_gradient_logreg(self.weights[:,i-1], X[:,i], y[i])
            else:
                print("Works only with linear and logistic regression!")
            updated_lr = self.update_lr(i)
            self.weights[:,i] = self.weights[:,i-1] - updated_lr * grad
        return self.weights

def compute_gradient_linreg(weights, X, y):
    return 2*(X*np.transpose(X)*weights - X*y)

def compute_gradient_logreg(w, X, y):
    return -y*X/(1+np.exp(y*np.dot(np.transpose(w), X))) 
    
if __name__ == '__main__':
    pass