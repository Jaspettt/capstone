import numpy as np

# Activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Input data
X = np.array([[0.15, 0.3, 0.45, 0.6],  # Dog image example (simplified)
              [0.2, 0.4, 0.6, 0.8]])  # Cat image example (simplified)

y = np.array([[1], [0]])

# Weights and biases
W1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
               [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
               [2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2]])
b1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])

W2 = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
               [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
               [2.6, 2.7, 2.8, 2.9, 3.0, 3.1],
               [3.2, 3.3, 3.4, 3.5, 3.6, 3.7],
               [3.8, 3.9, 4.0, 4.1, 4.2, 4.3],
               [4.4, 4.5, 4.6, 4.7, 4.8, 4.9]])
b2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])

W3 = np.array([[0.2, 0.3, 0.4, 0.5],
               [0.6, 0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6, 1.7],
               [1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5]])
b3 = np.array([[0.1, 0.2, 0.3, 0.4]])

W4 = np.array([[0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7],
               [0.8, 0.9, 1.0],
               [1.1, 1.2, 1.3]])
b4 = np.array([[0.1, 0.2, 0.3]])

W5 = np.array([[0.2], [0.3], [0.4]])
b5 = np.array([[0.1]])

# Forward propagation
z1 = np.dot(X, W1) + b1
a1 = leaky_relu(z1)

z2 = np.dot(a1, W2) + b2
a2 = leaky_relu(z2)

z3 = np.dot(a2, W3) + b3
a3 = leaky_relu(z3)

z4 = np.dot(a3, W4) + b4
a4 = leaky_relu(z4)

z5 = np.dot(a4, W5) + b5
a5 = sigmoid(z5)

# Compute required values
a5_values = a5.tolist()
a4_max = np.round(a4.max(axis=1)).tolist()
a3_max = np.round(a3.max(axis=1)).tolist()
a2_min = np.round(a2.min(axis=1)).tolist()
a1_min = np.round(a1.min(axis=1)).tolist()

# General conclusion
final_predictions = np.round(a5).astype(int).flatten()
conclusion = "Predicts image of cat" if final_predictions[0] == 0 else "Predicts image of dog"

print(a5_values)
print(a4_max)
print(a3_max)
print(a2_min)
print(a1_min)
print(conclusion)

# a5        =   [1 ,1]
# a4.max()  =   2955
# a3.max()  =   933
# a2.min()  =   71 
# a1.min()  =   3
# General Conclusion:  image of dog