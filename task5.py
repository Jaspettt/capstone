import numpy as np
import pandas as pd

df = pd.read_csv('./data/Question3_Final_CP 14.csv')
# Split dataset into features (X) and target variable (y)
X = df.drop(columns=["Target"])
y = df["Target"]

# Split dataset: 60% training, 40% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Train Logistic Regression model
clf_lr = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial', solver='lbfgs')
clf_lr.fit(X_train, y_train)

# Predictions
y_pred = clf_lr.predict(X_test)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

# Round results to 3 decimal places
accuracy_rounded = round(accuracy, 3)
f1_rounded = [round(score, 3) for score in f1]

accuracy_rounded, f1_rounded

# Accuracy:                 0.742
# F-1 score (class = 0):    0.701
# F-1 score (class = 1):    0.819
# F-1 score (class = 2):    0.756 ​