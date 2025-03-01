### Mean
```Python
# Select the 'Corruption level Rating Score (0-100)' column
sample_scores = df['Corruption level Rating Score (0-100)']

# Compute the sample mean
sample_mean = sample_scores.mean()
print(f"Mean of the selected corruption scores: {sample_mean:.2f}")
```
### Standard Error
![[Pasted image 20250224222304.png]]

```Python
# Select the 'Corruption level Rating Score (0-100)' column
sample_scores = df['Corruption level Rating Score (0-100)']

# Compute the sample mean
sample_mean = sample_scores.mean()
print(f"Mean of the selected corruption scores: {sample_mean:.2f}")
```
### Confidence interval
![[Pasted image 20250224222345.png]]

```Python

lower, upper = st.t.interval(confidence=0.95, df=len(df['Corruption level Rating Score (0-100)'])-1, 
              loc=mean,
              scale=standard_error_srs)

print(lower, upper)
```

**OR**
```Python
# t-value for 95% confidence interval (given as 2.04)
t_value = 2.04

# Compute the margin of error
margin_of_error = t_value * standard_error

# Compute the confidence interval
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error
print(f"95% Confidence Interval: ({lower_bound:f}, {upper_bound:f})")
```

### Stratum Weight
![[Pasted image 20250224222718.png]] 
Example: 
h - North America
$N_h = 8$ (seen from dataset)
$N = 32$
then $W_h = 0.25$
### Standard error of stratified part

#### Squared deviations from the mean
![[Pasted image 20250224222124.png]]
```Python
df['Y'] = (df['Corruption level Rating Score (0-100)']-52.28)**2
df['Y']
```
#### Stratum Variance estimates 
![[Pasted image 20250224223412.png]]
```Python
sh_1 = df[df['Stratum']=='North America']['Y'].sum() / 7
sh_2 = df[df['Stratum']=='South America']['Y'].sum() / 7
sh_3 = df[df['Stratum']=='Europe']['Y'].sum() / 7
sh_4 = df[df['Stratum']=='Central Asia']['Y'].sum() / 7

print(sh_1)
print(sh_2)
print(sh_3)
print(sh_4)

arr = [sh_1, sh_2, sh_3, sh_4]
arr
```
#### Standard error calculationf
![[Pasted image 20250224223541.png]]
```Python
import math
standart_strat = math.sqrt(((0.25**2*sh_1/8)+(0.25**2*sh_2/8)+(0.25**2*sh_3/8)+(0.25**2*sh_4/8)))
round(standart_strat, 2)
```
### D value
![[Pasted image 20250224223920.png]]

### N effective
![[Pasted image 20250224224156.png]]

### Standard error for Clustering random sampling
#### Variance per cluster
![[Pasted image 20250224225606.png]]
```Python
shc_1 = df[df['Cluster'] == 'Canada']['Y'].sum() / 3
shc_2 = df[df['Cluster'] == 'USA']['Y'].sum() / 3
shc_3 = df[df['Cluster'] == 'Columbia']['Y'].sum() / 3
shc_4 = df[df['Cluster'] == 'Brazil']['Y'].sum() / 3
shc_5 = df[df['Cluster'] == 'Spain']['Y'].sum() / 3
shc_6 = df[df['Cluster'] == 'France']['Y'].sum() / 3
shc_7 = df[df['Cluster'] == 'Uzbekistan']['Y'].sum() / 3
shc_8 = df[df['Cluster'] == 'Kazakhstan']['Y'].sum() / 3
```
#### Clusters Standard error
![[Pasted image 20250224225711.png]]
![[Pasted image 20250224225721.png]]
```Python
SE_with_clustering = (0.125**2*shc_1/4) + (0.125**2*shc_2/4) + (0.125**2*shc_3/4) + (0.125**2*shc_4/4) + (0.125**2*shc_5/4) + \
(0.125**2*shc_6/4) + (0.125**2*shc_7/4) + (0.125**2*shc_8/4)

SE_with_clustering = math.sqrt(SE_with_clustering)
print(round(SE_with_clustering, 2))
```

#### Rho approximation
![[Pasted image 20250224230042.png]]

#### N effective
![[Pasted image 20250224230423.png]]
### Normal equation method
```Python
import numpy as np  
  
# Добавляем столбец единичных значений для учета theta_0  
X = data[['X1', 'X2', 'X3', 'X4']].values  
X = np.c_[np.ones(X.shape[0]), X] # Добавляем столбец с единицами  
  
# Целевая переменная Y  
Y = data['Y'].values  
  
# Применяем метод нормального уравнения: theta = (X.T @ X)^(-1) @ X.T @ Y  
theta = np.linalg.inv(X.T @ X) @ X.T @ Y  
  
# Округляем параметры до трех знаков после запятой  
theta_rounded = np.round(theta, 3)  
theta_rounded
```

### Neural network
```Python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Define the neural network structure
input_size = 4
hidden_layer1_size = 5
hidden_layer2_size = 4
output_size = 1

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_layer1_size)
W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size)
W3 = np.random.randn(hidden_layer2_size, output_size)
b1 = np.zeros((1, hidden_layer1_size))
b2 = np.zeros((1, hidden_layer2_size))
b3 = np.zeros((1, output_size))

# Example training data (1 image of dog and 1 image of cat)
X = np.array([[0.2, 0.4, 0.6, 0.8],  # Example input for dog
              [0.1, 0.3, 0.5, 0.7]])  # Example input for cat
y = np.array([[1], [0]])  # Labels: 1 for dog, 0 for cat

# Training parameters
epochs = 100000
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    # Compute loss
    loss = np.mean((y - a3) ** 2)

    # Backpropagation
    d_a3 = 2 * (a3 - y)
    d_z3 = d_a3 * sigmoid_derivative(a3)
    d_W3 = np.dot(a2.T, d_z3)
    d_b3 = np.sum(d_z3, axis=0, keepdims=True)

    d_a2 = np.dot(d_z3, W3.T)
    d_z2 = d_a2 * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Update weights and biases
    W1 -= learning_rate * d_W1
    W2 -= learning_rate * d_W2
    W3 -= learning_rate * d_W3
    b1 -= learning_rate * d_b1
    b2 -= learning_rate * d_b2
    b3 -= learning_rate * d_b3

    # Print progress every 10,000 epochs
    if epoch % 10000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions
a3_final = (a3 > 0.5).astype(int)
print("Final Prediction:", "Dog" if a3_final[0] == 1 else "Cat")
print("a3:", a3)  # Final output of the neural network
print("a2.min():", a2.min())  # Minimum value of activations in second hidden layer
print("d_W1.min():", d_W1.min())  # Minimum value of weight gradients for first layer
```
### Pearson correlation coefficient
![[Pasted image 20250224231833.png]]
```Python
from scipy.stats import pearsonr
df_10 = pd.read_excel('datasets/quiz1_question10 (1).xlsx')

corr, _ = pearsonr(df_10['X'], df_10['Y'])
corr
```

### Missing values
```Python
import pandas as pd

# Load dataset from Google Drive link (you need to download it manually or use Google Colab)
file_path = 'path_to_your_downloaded_file.xlsx'  # Replace with the actual file path
df = pd.read_excel(file_path)

# Replace empty spaces or NaN with None for better handling
df = df.where(pd.notnull(df), None)

# Convert missing values to 0 and present values to 1
binary_mask = df.notnull().astype(int)

# Create a string pattern for each row (e.g., "1111", "1010")
df['_pattern'] = binary_mask.astype(str).agg(''.join, axis=1)

# Count frequency of each pattern
pattern_counts = df['_pattern'].value_counts().reset_index()
pattern_counts.columns = ['_pattern', '_freq']

# Calculate missing value count (_mv) for each pattern
pattern_counts['_mv'] = pattern_counts['_pattern'].apply(lambda x: x.count('0'))

# Display results
print(pattern_counts)

```
### Creating new values based on condition
```Python
df['w_1'] = np.where(df['Corruption level Rating Score (0-100)'] >= 50, 1, 0)
print(round(df['w_1'].mean(), 2))
```

```Python
df['w_2'] = np.where(df['Corruption level Rating Score (0-100)'] > 70, 1,
              np.where(df['Corruption level Rating Score (0-100)'] > 30, 0.5, 0))

print(round(df['w_2'].mean(), 2))
```


### Data Analysis theory questions
1. What is the first step in analyzing survey data as per the book? [Defining research questions]
2. What does a probability sampling plan ensure in a survey? [Known non‑zero probability for inclusion]
3. Which of the following is not addressed in the book [Applied Survey Data Analysis] as a non-probability sampling method? (P.S please indicate the name of sampling) [Model‑dependent sampling]
4.  What is a key benefit of stratification in survey design? [It reduces standard errors for estimates.]
5. What is the significance of the finite population correction (FPC) in sampling? [It reduces overestimation of variance for small sampling fractions.]
6. What is the "design effect" in complex sampling? [The increased variance due to clustering and weighting]
7. What is the role of weighting in survey data analysis? [It adjusts for unequal probabilities of selection.]
8.  How is the nonresponse adjustment factor calculated in surveys? [By using inverse probability of response propensity]
9. Which statistical method is commonly used for modeling response propensities in surveys? [Logistic regression]
10. Which survey design feature typically increases standard errors in estimates? [Clustering]
11. What is the primary responsibility of a **data producer** in survey data preparation? [To develop sampling weights and provide a cleaned data set]
12. Which factor is **not** part of the final analysis weight provided in survey datasets? [Stratification weight]
13. Why is it important to assess the distribution of weight variables before analysis? [To identify scaling, skew, and extreme weight values]
14. What should an analyst do first when exploring the rates and patterns of missing data in Stata? [Use the mvpatterns command]
15. What is the consequence of ignoring missing data in survey analysis? [Bias in parameter estimates]
16. What approach is recommended for subpopulation analysis in complex surveys? [Unconditional subclass analysis]
17. Why is collapsing strata used in sampling error calculation models? [To mask original strata for confidentiality]
18. What is a critical step before analyzing a new survey data set? [Identifying weight variables and their scaling]
19. Which method is **not** mentioned as a statistical imputation strategy? [Maximum likelihood imputation]
20. What tool in Stata helps set up survey design variables for sampling error estimation? [svyset]

### Machine learning theory

1. What is the primary requirement for testing the independence of two categorical variables in survey data? [The Rao‑Scott correction must be applied]
2. hat is the design effect (“DEFF”) used for in survey analysis?  [To adjust standard errors for complex survey designs]
3. Which statistical method is used to estimate proportions for a binary survey variable?  [Taylor Series Linearization (TSL)]
4. How are confidence intervals for proportions adjusted when they are close to 0 or 1? [By using a logit transformation]
5. What is a multinomial categorical variable?  [A categorical variable with more than two levels]
6. What is the primary goal of the Cochran-Mantel-Haenszel test?  [To test the association between two variables while controlling for a third]
7. What statistic is used in the Rao-Scott chi-square test for categorical data? [Pearson Chi‑Square Test Statistic]
8.  Which software is mentioned for the analysis of categorical data in survey research? [Stata]
9. Which graphical technique is most effective for displaying proportions of categorical survey variables? [Pie Charts and Bar Charts]
10.  In Example 6.2 (Chapter 6.3.2), which ethnicity group had the largest proportion in the NHANES dataset? [White]
11.  What is the primary goal of supervised learning in regression problems? [Predict real‑valued outputs]
12. What is the purpose of the cost function J(θ0,θ1)?  [To minimize the residual sum of squares]
13.  What does the term “batch gradient descent” mean? [It updates parameters using all training examples]
14. What happens when the learning rate α is too small? [Gradient descent slows down significantly]
15. What is the role of the parameter θ0 in linear regression? [It represents the intercept of the line]
16. Which metric does gradient descent minimize to find the optimal parameters? [Mean Squared Error (MSE)]
17. In the training set for housing prices, what does m represent? [The number of training examples]а
18. What is the intuition behind the cost function in linear regression? [It measures the distance between actual and predicted values]
19. What is the primary purpose of logistic regression in classification problems? [To model the probability of a binary outcome]  

2) What is the purpose of the decision boundary in logistic regression? [To classify data points based on a threshold]

3) How is the cost function of logistic regression designed to handle classification tasks? [ It uses the cross‑entropy loss function]

4) What does gradient descent aim to achieve in the training of a logistic regression model? [It finds the parameters that minimize the cost function]

5) Which optimization algorithm is often used as an alternative to gradient descent for logistic regression? [Newton’s Method]

1) Why is the cost function for logistic regression designed as a log-likelihood function rather than a squared error function?  [Because squared error leads to a non‑convex cost function in logistic regression]  

2) Given a logistic regression model with parameters theta, which statement best describes how the decision boundary is formed?  [It is determined by the equation where hypothesis = 0,5]

3) When performing logistic regression with gradient descent, what is the primary challenge when using a very large dataset?  [The gradient update step becomes computationally expensive]

4) Suppose you are training a logistic regression model and notice that it has high variance. What is the most effective approach to address this issue?  [Apply L2 regularization to reduce overfitting]

5) Why do we use the "One-vs-All" (OvA) method in logistic regression for multi-class classification? [Because logistic regression inherently supports only binary classification]

1) Which of the following best describes the reason why neural networks outperform traditional machine learning models in complex pattern recognition tasks? [They can fit any function given enough hidden units and training data.]

2) What is the significance of the “one learning algorithm” hypothesis in neural networks? [It implies that the same neural mechanisms underlie different types of learning in biological and artificial systems.]

3) Why do modern deep learning architectures prefer using ReLU (Rectified Linear Unit) over sigmoid activation functions in hidden layers? [ReLU prevents vanishing gradients, allowing deeper networks to train effectively.]

4) What is the primary purpose of vectorized implementation in forward propagation for neural networks? [To reduce computational complexity by avoiding iterative loops.]

5) Why is weight initialization important in training deep neural networks? [Poor initialization can lead to vanishing or exploding gradients.]

1) Which of the following statements best describes how neural networks handle non-linearly separable problems like XOR? [A non‑linear activation function and multiple hidden layers enable networks to model XOR.]

2) In multi-class classification using a neural network, which approach is typically used to handle multiple output classes? [Using the softmax function in the output layer.]

3) What is the primary advantage of using convolutional neural networks (CNNs) instead of fully connected networks for image classification tasks? [CNNs use weight‑sharing mechanisms that significantly reduce the number of parameters.]

4) Which optimization algorithm is most commonly used to minimize the loss function in neural networks?  [Gradient Descent]
### Precision, Accuracy, Recall, F1 Score
![[Pasted image 20250225100703.png]]

![[Pasted image 20250225100722.png]]
![[Pasted image 20250225100738.png]]
![[Pasted image 20250225100747.png]]

```Python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Define confusion matrix
confusion_matrix = np.array([[50, 30, 0],
                             [10, 20, 2],
                             [8, 10, 30]])

# Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
TP = np.diag(confusion_matrix)
FP = np.sum(confusion_matrix, axis=0) - TP
FN = np.sum(confusion_matrix, axis=1) - TP

# Compute Precision, Recall, and F1-Score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)

# Compute Accuracy
total_correct = np.sum(TP)
total_samples = np.sum(confusion_matrix)
accuracy = total_correct / total_samples

# Print results
print(f"Accuracy: {accuracy:.3f}")
print("Class-wise metrics:")
for i, class_label in enumerate(['a', 'b', 'c']):
    print(f"Class {class_label}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1-score={f1[i]:.3f}")
```

### Testing theory
This combines data from one module with all other modules to test data flow between them → Non-Incremental Testing,

This non-functional testing ensures the software operates correctly across different hardware, operating systems, applications, network environments, or mobile devices. → Compatability Testing,

Also known as User Experience Testing, it checks how user-friendly and easy to use the software is → Usability Testing,

This assesses the speed, response time, stability, reliability, scalability, and resource usage of the software under specific workloads. → Performance Testing,

This ensures the application performs well consistently over an acceptable period. → Stability Testing,

This testing method examines the internal structure, design, data structures, code, and functionality of the software. → White Box Testing,

This involves logically integrating and testing multiple software modules as a group. → Integration Testing,

  
This ensures the software system meets the specified functional requirements → Functional Testing,

This verifies the software’s stability and reliability under extreme conditions. → Stress Testing,

This evaluates non-functional aspects such as reliability, performance, load handling, and accountability of the software. → Non-Functional Testing,

This validates the fully integrated software product. → System Testing,

This approach tests the software’s functionalities without any knowledge of its internal code, paths, or implementation details. → Black Box Testing,

This type of testing focuses on individual units or components of the software. → Unit Testing,

This measures the system’s performance as the number of user requests increases or decreases → Scalability Testing,

This approach tests higher-level modules first, using stubs for any undeveloped submodules → Top-Down Testing,

This evaluates the software’s performance under expected load conditions → Load Testing,

This method integrates modules one by one to identify defects during development → Incremental Testing,

This method involves partial knowledge of the internal structure of the application, aiming to identify defects caused by improper code or application usage. → Grey Box Testing,

This involves testing software based on the client’s requirements without the use of any automation tools. → Manual Testing,

This method tests lower-level modules first, using test drivers to pass data from higher to lower levels. → Bottom-Up Testing,

This refers to testing software using automation tools to meet the client’s needs. → Automation Testing

#### Capture recapture method
![[Pasted image 20250225114624.png]]

![[Pasted image 20250225114633.png]]
![[Pasted image 20250225114639.png]]

