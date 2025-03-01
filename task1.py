import numpy as np
import pandas as pd

# Load the dataset
file_path = "./data/Question1_Final_CP 14.xlsx"
df = pd.read_excel(file_path)

# Extract relevant variable
variable_values = df['Variable']

# Compute SRS mean
mean_srs = round(np.mean(variable_values), 2)

# Compute SRS standard error
std_dev = np.std(variable_values, ddof=1)  # Sample standard deviation
n = len(variable_values)  # Sample size
std_error_srs = round(std_dev / np.sqrt(n), 2)

# Compute 95% confidence interval for SRS
t_value = 2.04
margin_of_error = round(t_value * std_error_srs, 2)
upper_limit_srs = round(mean_srs + margin_of_error, 2)
lower_limit_srs = round(mean_srs - margin_of_error, 2)

# Compute stratified sampling parameters
strata = df['Stratum'].unique()
Nh = df.groupby('Stratum').size()  # Number of observations per stratum
N = len(df)  # Total sample size
Wh = Nh / N  # Stratum weights

# Compute mean per stratum
mean_h = df.groupby('Stratum')['Variable'].mean()

# Compute overall stratified mean
mean_stratified = round(sum(Wh * mean_h), 2)

# Compute stratified standard error
variance_h = df.groupby('Stratum')['Variable'].var(ddof=1)  # Variance per stratum
std_error_stratified = round(np.sqrt(sum((Wh ** 2) * (variance_h / Nh))), 2)

# Compute d-value and d-squared
d_value = round(std_error_stratified / std_error_srs, 2)
d_squared = round(d_value ** 2, 2)

# Compute Neff (effective sample size)
Neff = round(N * d_squared, 2)

# Print results
print(f"SRS Mean: {mean_srs}")
print(f"SRS Standard Error: {std_error_srs}")
print(f"SRS 95% CI Upper Limit: {upper_limit_srs}")
print(f"SRS 95% CI Lower Limit: {lower_limit_srs}")
print(f"Stratum Weights (Wh): {Wh.values}")
print(f"Stratified Mean: {mean_stratified}")
print(f"Stratified Standard Error: {std_error_stratified}")
print(f"d-value: {d_value}")
print(f"d-squared: {d_squared}")
print(f"Neff: {Neff}")

# SRS Mean:                     46.1
# SRS Standard Error:           5.83
# SRS 95% CI Upper Limit:       57.99
# SRS 95% CI Lower Limit:       34.21
# Stratum Weights (Wh):         [0.25 0.25 0.25 0.25]
# Stratified Mean:              46.1
# Stratified Standard Error:    5.7
# d-value:                      0.98
# d-squared:                    0.96
# Neff:                         15.36