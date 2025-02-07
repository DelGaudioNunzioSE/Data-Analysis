import numpy as np
from sgd_class import StochasticGradientDescent
import random
import matplotlib.pyplot as plt
from scipy.stats import norm


#sigma_easy=0.7, sigma_hard=2
def likelihood(x, mu, sigma):
    l = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-((x - mu)**2) / (2*sigma**2))
    return l

def posterior(x,mu1,mu2,sigma,pi1,pi2):
    l1 = likelihood(x,mu1,sigma) #bisogna passare radice quadrata
    l2 = likelihood(x,mu2,sigma)
    return l1*pi1/((l1*pi1) + (l2*pi2))

def hand_posterior(x,sigma):
    ps = 1/(1+np.exp(-2*x/sigma))
    return ps

def likelihood_ratio(l1,l2):
    return np.log(l1/l2)

#facciamo il predict con map utilizzando il log del rapporto tra le due likelihood
def predict_MAP(mu_pos, mu_neg, sigma, pi1, pi2, x_test, y_test):
    predictions = []
    for x in x_test:
        lr = np.log(likelihood(x,mu_pos,sigma) / likelihood(x,mu_neg,sigma))
        threshold = np.log(pi1 / pi2)
        prediction = 1 if lr > threshold else -1
        predictions.append(prediction)
    
    correct_predictions = sum(p == y for p, y in zip(predictions, y_test))
    accuracy = correct_predictions / len(y_test)
    return predictions, accuracy.item()
    
def gen_data(mu_pos,mu_neg,sigma,n_samples):
    n_samples = n_samples
    p_pos = 0.5           # prior probability of the hypothesis +1
    p_neg = 1 - p_pos     # prior probability of the hypothesis -1
    y = np.random.choice(np.array([-1,1]), (n_samples,1), p=[p_neg, p_pos]) #-1 con prob p_neg e +1 con prob p_pos
    n_features = 1
    X = np.zeros((n_features,n_samples))
    mu_pos = mu_pos
    mu_neg = mu_neg
    sigma = sigma
    for i in range(n_samples):
        X[:,i] = np.random.normal(mu_pos, sigma) if y[i] == 1 else np.random.normal(mu_neg, sigma)
    X = np.vstack((X, np.ones((1,n_samples))))
    return X,y


def compute_gradient_logreg(w, X, y):
    return -y*X/(1+np.exp(y*np.dot(np.transpose(w), X)))

def compute_classifier_accuracy(w, X, y):
    return np.mean(np.sign(np.dot(np.transpose(w), X)).T == y)

def sgd(n_features, n_samples, mu_pos, mu_neg, sigma, n_montecarlo, step = True, lr=0.05):
    sgd_logreg = StochasticGradientDescent(learning_rate=lr, constant_step_size=step)
    weights_logreg_mc = np.zeros((n_features+1, n_samples)) #+1 per B0
    for i in range(n_montecarlo):
        X,y = gen_data(mu_pos, mu_neg, sigma,n_samples)
        tmp_weights_logreg = sgd_logreg.run_sgd(X,y,'logloss')

        weights_logreg_mc += tmp_weights_logreg
    weights_logreg_mc = weights_logreg_mc/n_montecarlo
    return weights_logreg_mc


if __name__ == '__main__':

    #Quesito 1 e 2
    #abbiamo scelto una sigma2_easy tale che le gaussiane riferite alle due probabilità non si intersecano
    #viceversa abbiamo scelto una sigma2_diff tale che i valori delle due gaussiane si intersecano e ciò rende la
    #scelta più complessa, in quanto la differenza tra classi non è così evidente.
    random.seed(2022)
    sigma2_easy = 0.8
    sigma2_diff = 3
    x_values = np.linspace(-3, 3, 100)
    posterior_easy = [posterior(x,1,-1,np.sqrt(sigma2_easy),1/2,1/2) for x in x_values]
    posterior_diff = [posterior(x,1,-1,np.sqrt(sigma2_diff),1/2,1/2) for x in x_values]
    posterior_hand_easy = [hand_posterior(x,sigma2_easy) for x in x_values]
    posterior_hand_diff = [hand_posterior(x,sigma2_diff) for x in x_values]

    #plot con la formula iniziale di P[1|x]
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, posterior_easy, label=f'Easy: $\sigma^2={sigma2_easy}$', color='blue')
    plt.plot(x_values, posterior_diff, label=f'Difficult: $\sigma^2={sigma2_diff}$', color='red')
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.7)
    plt.axvline(0, linestyle='--', color='black', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('$P(Y=+1 | X=x)$')
    plt.legend()
    plt.title('Posterior Probability $P(Y=+1 | X=x)$ for different variances')
    plt.show()

    #plot con la formula posterior calcolata da noi di P[1|x]
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, posterior_hand_easy, label=f'Easy: $\sigma^2={sigma2_easy}$', color='orange')
    plt.plot(x_values, posterior_hand_diff, label=f'Difficult: $\sigma^2={sigma2_diff}$', color='green')
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.7)
    plt.axvline(0, linestyle='--', color='black', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('$P(Y=+1 | X=x)$')
    plt.legend()
    plt.title('Hand Posterior Probability $P(Y=+1 | X=x)$ for different variances')
    plt.show()

    #i due plot sono uguali

    posterior_easy_neg = [norm.pdf(x, loc=-1, scale=sigma2_easy) for x in x_values]
    posterior_diff_neg = [norm.pdf(x, loc=-1, scale=sigma2_diff) for x in x_values]

    posterior_easy_pos = [norm.pdf(x, loc=1, scale=sigma2_easy) for x in x_values]
    posterior_diff_pos = [norm.pdf(x, loc=1, scale=sigma2_diff) for x in x_values]

    #plot delle pdf delle gaussiane nei due casi
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, posterior_easy_neg, label=f'Easy Neg:', color='blue')
    plt.plot(x_values, posterior_easy_pos, label=f'Easy Pos:', color='red')
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.7)
    plt.axvline(0, linestyle='--', color='black', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.title('Pdf x Sigma easy ')
    plt.show()
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, posterior_diff_neg, label=f'Difficult Nos', color='blue')
    plt.plot(x_values, posterior_diff_pos, label=f'Difficult Pos', color='red')
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.7)
    plt.axvline(0, linestyle='--', color='black', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.legend()
    plt.title('Pdf x Sigma Hard')
    plt.show()


    #Quesito 3
    mci = 5
    accuracy_map_easy = np.zeros((mci))
    accuracy_map_diff = np.zeros((mci))
    
    #facciamo generare nuovi test dentro montecarlo ogni volta
    for mc in range(mci):
        x_test_easy, y_test_easy = gen_data(1,-1,np.sqrt(sigma2_easy),1000)
        x_test_diff, y_test_diff = gen_data(1,-1,np.sqrt(sigma2_diff),1000)
        _,accuracy_map_easy[mc] = predict_MAP(1,-1,np.sqrt(sigma2_easy),1/2,1/2,x_test_easy[0,:],y_test_easy)
        _,accuracy_map_diff[mc] = predict_MAP(1,-1,np.sqrt(sigma2_diff),1/2,1/2,x_test_diff[0,:],y_test_diff)

    print(f"Accuracy MAP EASY {np.mean(accuracy_map_easy)}") #90%
    print(f"Accuracy MAP DIFF {np.mean(accuracy_map_diff)}") #70%

    print(f"ErrorProb MAP EASY {1-np.mean(accuracy_map_easy)}") #10%
    print(f"ErrorProb MAP DIFF {1-np.mean(accuracy_map_diff)}") #30%

    #caso2
    accuracy_sgd = np.zeros((mci)) #test con learning rate fisso
    accuracy_sgd_lr = np.zeros((mci)) #test con learning rate variabile

    weights= sgd(1,1000,1,-1,np.sqrt(sigma2_easy),10) #pesi lr fisso
    weights_lr= sgd(1,1000,1,-1,np.sqrt(sigma2_easy),10, False, 1) #pesi lr variabile
    for mc in range(mci):
        x_test_sgd, y_test_sgd = gen_data(1,-1,np.sqrt(sigma2_easy),1000)
        accuracy_sgd[mc] = compute_classifier_accuracy(weights[:,-1].reshape(-1,1), x_test_sgd, y_test_sgd)
        accuracy_sgd_lr[mc] = compute_classifier_accuracy(weights_lr[:,-1].reshape(-1,1), x_test_sgd, y_test_sgd)

    print(f"Accuracy SGD EASY {np.mean(accuracy_sgd)}")
    print(f"ErrorProb SGD EASY {1-np.mean(accuracy_sgd)}") 

    print(f"Accuracy SGD_LR EASY {np.mean(accuracy_sgd_lr)}") 
    print(f"ErrorProb SGD_LR EASY {1-np.mean(accuracy_sgd_lr)}")

    #plot dei pesi finali
    plt.figure()
    plt.plot(weights[0,:], label=r"$\beta_1$")
    plt.plot(weights[1,:], label=r"$\beta_2$")
    plt.xlabel("Iterations")
    plt.ylabel("Weights evolution")
    plt.legend()
    plt.grid(True)
    plt.show() 






