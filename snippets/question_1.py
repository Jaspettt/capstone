import pandas as pd
import numpy as np
import math
df = pd.read_excel("Question1_Final_CP.xlsx")


variable_score = df["Variable"]
sample_mean = variable_score.mean()

sample_std_dev = variable_score.std(ddof=1)  # Use ddof=1 for sample standard deviation, not for population)

# Compute the sample size (n)
sample_size = len(variable_score)

# Compute the standard error (SE)
standard_error = sample_std_dev / np.sqrt(sample_size)
print(f"Standard error: {standard_error:.2f}")
t_value = 2.04

# Compute the margin of error
margin_of_error = t_value * standard_error

# Compute the confidence interval
lower_bound = (sample_mean - margin_of_error)
upper_bound = (sample_mean + margin_of_error)
# print(f"95% Confidence Interval: ({lower_bound:.2f}, {upper_bound:.2f})")
print("95% Confidence Interval:", lower_bound, upper_bound)
# Calculate the total population size
N = len(df)

# Group by Stratum
strata_group = df.groupby('Cluster')

# Compute the mean and size for each stratum
stratum_means = strata_group['Variable'].mean()
stratum_sizes = strata_group.size()

# Compute the weight for each stratum (Wh)
stratum_weights = stratum_sizes / N

# Output results
print(f"Stratum Weights (W_h):\n{stratum_weights}")
print("Compute a mean:", df['Variable'].mean().round(2))
df['Y'] = (df['Variable']-52.28)**2

stratums = df['Cluster'].unique()
sh_values = {}

for stratum in stratums:
    sh_values[stratum] = df[df['Cluster'] == stratum]['Y'].sum() / int(len(df[df['Cluster'] == stratum]['Y']) - 1)

#4.3
print(sh_values)
standart_strat = math.sqrt(((0.125**2*sh_values[1]/2)+
                            (0.125**2*sh_values[2]/2)+
                            (0.125**2*sh_values[3]/2)+
                            (0.125**2*sh_values[4]/2)+
                            (0.125**2*sh_values[5]/2)+
                            (0.125**2*sh_values[6]/2)+
                            (0.125**2*sh_values[7]/2)+
                            (0.125**2*sh_values[8]/2)))

print("Compute a standard error for Stratified part:", standart_strat)