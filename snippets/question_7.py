import pandas as pd
import numpy as np

df = pd.read_excel("Question7_Final_CP.csv")

sample_scores = df['Corruption level Rating Score (0-100)']

# Compute the sample mean
sample_mean = sample_scores.mean()
print(f"Mean of the selected corruption scores: {sample_mean:.2f}")