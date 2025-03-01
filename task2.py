import numpy as np
import pandas as pd

# Load the dataset
file_path = "Question1_Final_CP 14.xlsx"
df = pd.read_excel(file_path, sheet_name='Лист1')

# Hypothesis function:
# h_theta(X) = theta_0 + theta_1*X1 + theta_2*X2 + theta_3*X1^2 + theta_4*X1^3 + theta_5*X2^2 + theta_6*X2^3 + theta_7*(X1*X2) + theta_8*(X1^2 * X2)

# Adjusting the feature matrix according to the given hypothesis
df['X1*X2'] = df['X1'] * df['X2']
df['X1^2*X2'] = (df['X1'] ** 2) * df['X2']

# Selecting the required features based on the new hypothesis
features_adjusted = df[['X1', 'X2', 'X1^2', 'X1^3', 'X2^2', 'X2^3', 'X1*X2', 'X1^2*X2']]

# Normalize the selected features
normalized_features_adjusted = (features_adjusted - features_adjusted.mean()) / features_adjusted.std()

# Add bias term (column of ones)
X_adjusted = np.c_[np.ones(normalized_features_adjusted.shape[0]), normalized_features_adjusted]

# Extract output feature
y = df['Y'].values.reshape(-1, 1)

# Initialize theta parameters to zero
theta = np.zeros((X_adjusted.shape[1], 1))

# Define gradient descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        theta -= alpha * gradients
        cost = (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)
        cost_history.append(round(cost))  # Round to nearest integer

    return theta, cost_history

# Run gradient descent for different iterations
iterations_list = [100, 1000, 10000]
results_adjusted = {}

for n in iterations_list:
    theta_opt, cost_history = gradient_descent(X_adjusted, y, np.zeros((X_adjusted.shape[1], 1)), 0.01, n)
    max_theta_value = int(round(np.max(theta_opt)))  # Round to nearest integer
    results_adjusted[n] = (cost_history[-1], theta_opt.flatten(), max_theta_value)

# Print updated results
for n, (cost, theta_vals, max_theta) in results_adjusted.items():
    print(f"Iterations: {n}")
    print(f"Cost Function: {cost}")
    print(f"Optimal Theta: {theta_vals}")
    print(f"Max Theta Value: {max_theta}\n")


# hz hypothesis ne ukazan, esli on takoy:
# h_theta(X) = theta_0 + theta_1*X1 + theta_2*X2 + theta_3*X1^2 + theta_4*X1^3 + theta_5*X2^2 + theta_6*X2^3 + theta_7*(X1*X2) + theta_8*(X1^2 * X2)
# to

#               Cost Function       Maximum Theta
# n=100         970010              2110
# n=1000        43132               3327
# n=10000       1264                3328