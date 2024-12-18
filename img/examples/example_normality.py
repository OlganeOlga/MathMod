import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example data
data = np.random.normal(loc=0, scale=1, size=1000)  # Normally distributed data

# Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot for Normal Distribution")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True)
plt.show()

# Create datasets for different cases
data_normal = np.random.normal(loc=0, scale=1, size=1000)  # Normal distribution
data_right_skewed = np.random.exponential(scale=1, size=1000)  # Right-skewed distribution
data_left_skewed = -np.random.exponential(scale=1, size=1000)  # Left-skewed distribution
data_heavy_tailed = np.random.standard_t(df=3, size=1000)  # Heavy-tailed (t-distribution)
data_light_tailed = np.random.beta(a=2, b=2, size=1000)  # Light-tailed distribution

# List of datasets and labels
datasets = [
    (data_normal, "Normal Distribution"),
    (data_right_skewed, "Right-Skewed Distribution"),
    (data_left_skewed, "Left-Skewed Distribution"),
    (data_heavy_tailed, "Heavy-Tailed Distribution"),
    (data_light_tailed, "Light-Tailed Distribution"),
]

# Plot Q-Q plots for each dataset
plt.figure(figsize=(12, 8))
for i, (data, title) in enumerate(datasets, start=1):
    plt.subplot(2, 3, i)  # Create a subplot grid
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(title)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
