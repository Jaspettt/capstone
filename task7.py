import pandas as pd
import numpy as np
from scipy.stats import t

# Load dataset
file_path = "./data/Question7 1.csv"
df = pd.read_csv(file_path)

# Count occurrences of each income level
counts = df['Income Level'].value_counts()
total_n = len(df)

# Prior count for smoothing
alpha = 2

# Compute estimated proportions
proportions = (counts + alpha) / (total_n + alpha * len(counts))

# Compute standard error
se = np.sqrt(proportions * (1 - proportions) / (total_n + alpha * len(counts)))

# Compute 95% confidence intervals using t-distribution
df_degrees = total_n - 1  # Degrees of freedom
t_critical = t.ppf(0.975, df_degrees)

ci_lower = proportions - t_critical * se
ci_upper = proportions + t_critical * se

# Compute Shannon Entropy Measure
entropy = -np.sum(proportions * np.log2(proportions))

# Round values
proportions_rounded = proportions.round(2)
se_rounded = se.round(3)
ci_lower_rounded = ci_lower.round(3)
ci_upper_rounded = ci_upper.round(3)
entropy_rounded = round(entropy, 3)

# Prepare results
results = {
    "Income Level": proportions.index.tolist(),
    "n": counts.tolist(),
    "Estimated Proportion": proportions_rounded.tolist(),
    "Standard Error": se_rounded.tolist(),
    "95% CI Lower": ci_lower_rounded.tolist(),
    "95% CI Upper": ci_upper_rounded.tolist(),
    "Entropy Measure": [entropy_rounded] * len(proportions)
}

# Convert to DataFrame and display
results_df = pd.DataFrame(results)
import ace_tools as tools
tools.display_dataframe_to_user(name="Income Level Analysis", dataframe=results_df)

# Income Level      n,      Estimated Proportion    Standard Error  95% CI Lower    95% CI Upper    Entropy Measure
# Middle            750     0.5                     0.013           0.474           0.525           1.479
# Low               467     0.31                    0.012           0.288           0.335           1.479
# High              283     0.19                    0.01            0.169           0.209           1.479