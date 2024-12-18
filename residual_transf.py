import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
# Temperaturdata för tre stationer
import log_transformation as l_t


# Temperaturdata för första stationen (Halmstad flygplats)
x = np.arange(l_t.data.shape[1])  # Tid (index)
y = l_t.data[0]  # Temperaturer

# Linjär modell residualer
y_pred = l_t.intercept + l_t.slope * x
residuals_linear = y - y_pred

# Transformerad modell residualer
y_log = np.log(y - np.min(y) + 1)
y_log_pred = l_t.intercept_log + l_t.slope_log * x
y_exp_pred = np.exp(y_log_pred) - 1 + np.min(y)
residuals_transformed = y - y_exp_pred

# Plotta residualer mot index
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x, residuals_linear, color='red', alpha=0.7, label="Linjär modell residualer")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Residualer för linjär modell")
plt.xlabel("Index")
plt.ylabel("Residualer")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(x, residuals_transformed, color='green', alpha=0.7, label="Transformerad modell residualer")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Residualer för transformerad modell")
plt.xlabel("Index")
plt.ylabel("Residualer")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plotta residualernas fördelning och jämför med normalfördelning
plt.figure(figsize=(12, 6))

# Linjär modell
plt.subplot(2, 2, 1)
sns.histplot(residuals_linear, kde=True, color='red', label="Linjär modell")
plt.title("Histogram av residualer (linjär modell)")
plt.xlabel("Residualer")
plt.legend()

plt.subplot(2, 2, 2)
probplot(residuals_linear, dist="norm", plot=plt)
plt.title("Q-Q Plott (linjär modell)")

# Transformerad modell
plt.subplot(2, 2, 3)
sns.histplot(residuals_transformed, kde=True, color='green', label="Transformerad modell")
plt.title("Histogram av residualer (transformerad modell)")
plt.xlabel("Residualer")
plt.legend()

plt.subplot(2, 2, 4)
probplot(residuals_transformed, dist="norm", plot=plt)
plt.title("Q-Q Plott (transformerad modell)")

plt.tight_layout()
plt.show()

# Varians av residualer
variance_linear = np.var(residuals_linear)
variance_transformed = np.var(residuals_transformed)

print(f"Varians för residualer (linjär modell): {variance_linear:.4f}")
print(f"Varians för residualer (transformerad modell): {variance_transformed:.4f}")
