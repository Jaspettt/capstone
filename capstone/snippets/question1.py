import numpy as np
import pandas as pd

# Load the dataset
file_path = "./Question1_Final_CP.xlsx"
xls = pd.ExcelFile(file_path)
data = pd.read_excel(xls, sheet_name="Лист1")

# Extract the relevant column
variable_data = data["Variable"]

# Simple Random Sampling (SRS) calculations
mean_srs = variable_data.mean()
std_dev = variable_data.std(ddof=1)  # Sample standard deviation
n = len(variable_data)  # Sample size
se_srs = std_dev / np.sqrt(n)

t_value = 2.04  # Given t-value for 95% CI
ci_lower_srs = mean_srs - t_value * se_srs
ci_upper_srs = mean_srs + t_value * se_srs

# Round results to 2 decimal places
mean_srs = round(mean_srs, 2)
se_srs = round(se_srs, 2)
ci_lower_srs = round(ci_lower_srs, 2)
ci_upper_srs = round(ci_upper_srs, 2)

# Stratified Sampling Calculations
stratum_counts = data["Stratum"].value_counts().sort_index()
total_n = len(data)
Wh = (stratum_counts / total_n).round(2).to_dict()

stratum_means = data.groupby("Stratum")["Variable"].mean()
mean_stratified = np.sum(pd.Series(Wh) * stratum_means)

stratum_vars = data.groupby("Stratum")["Variable"].var(ddof=1)
stratum_ns = data.groupby("Stratum")["Variable"].count()

se_stratified = np.sqrt(np.sum((pd.Series(Wh)**2) * (stratum_vars / stratum_ns)))

d_value = se_stratified / se_srs
d_squared = d_value ** 2
Neff = total_n / d_squared

# Round results to 2 decimal places
mean_stratified = round(mean_stratified, 2)
se_stratified = round(se_stratified, 2)
d_value = round(d_value, 2)
d_squared = round(d_squared, 2)
Neff = round(Neff, 2)

# Print results
print("Simple Random Sampling (SRS) Results:")
print(f"Mean (SRS): {mean_srs}")
print(f"Standard Error (SRS): {se_srs}")
print(f"95% Confidence Interval (SRS): ({ci_lower_srs}, {ci_upper_srs})\n")

print("Stratified Random Sampling (Stratified RS) Results:")
print(f"Stratum Weights (Wh): {Wh}")
print(f"Mean (Stratified RS): {mean_stratified}")
print(f"Standard Error (Stratified RS): {se_stratified}")
print(f"d-value (SE_stratified / SE_SRS): {d_value}")
print(f"d-squared: {d_squared}")
print(f"Neff (Effective Sample Size): {Neff}")
