from scipy.special import expit
import pandas as pd
import numpy as np

df_logistic = pd.read_csv('./data/Question3_Final_CP 14.csv')
# Normalize the input features
features_logistic = df_logistic[['X1', 'X2', 'X3']]
normalized_features_logistic = (features_logistic - features_logistic.mean()) / features_logistic.std()

# Add bias term (column of ones)
X_logistic = np.c_[np.ones(normalized_features_logistic.shape[0]), normalized_features_logistic]

# Extract output feature
y_logistic = df_logistic['Y'].values.reshape(-1, 1)

# Define logistic regression cost function with regularization
def compute_cost(X, y, theta, lambda_):
    m = len(y)
    h = expit(X @ theta)
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    return round(cost, 2)  # Round to 2 decimal places

# Define gradient descent function for logistic regression with regularization
def gradient_descent_logistic(X, y, theta, alpha, lambda_, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        h = expit(X @ theta)
        gradients = (1 / m) * X.T @ (h - y) + (lambda_ / m) * np.r_[[[0]], theta[1:]]
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta, lambda_)
        cost_history.append(cost)

    return theta, cost_history

# Run gradient descent for different iterations, learning rates, and lambda values
logistic_params = [
    (100, 1, 1),
    (1000, 1, 10),
    (10000, 2, 5)
]

results_logistic = {}

for n, alpha, lambda_ in logistic_params:
    theta_opt, cost_history = gradient_descent_logistic(X_logistic, y_logistic, np.zeros((X_logistic.shape[1], 1)), alpha, lambda_, n)
    max_theta_value = round(np.max(theta_opt), 2)  # Round to 2 decimal places
    results_logistic[n] = (cost_history[-1], theta_opt.flatten(), max_theta_value)

# Display updated results
print(results_logistic)


#                       Cost Function       Maximum Theta Value
# n=100,α=1,λ=1         0.17                2.00
# n=1000,α=1,λ=10       0.33                1.00
# n=10000,α=2,λ=5       0.27                1.34
# 4