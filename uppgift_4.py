"""
this file shows linniar regression
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, t

# Temperaturdata för tre stationer
data = np.array([
    [-2.4, -2.8, -2.9, -2.8, -2.8, -2.8, -3.3, -4.3, -2.1, -1.4, -0.7, -0.1,
     2.4, 2.7, 3.5, 3.8, 3.8, 3.9, 3.8, 3.4, 3.2, 2.6, 2.3, 2.0, 2.0],
    [-6.6, -7.1, -6.3, -6.2, -6.6, -6.3, -6.2, -6.1, -5.8, -5.5, -5.1, -4.3,
     -4.0, -3.3, -2.7, -1.7, -1.2, -1.7, -1.9, -1.7, -1.5, -1.6, -1.8, -1.8,
     -1.9],
    [-10.9, -10.6, -11.3, -12.2, -10.8, -9.0, -9.8, -8.4, -6.7, -5.7, -4.7, -3.6,
     -3.2, -2.7, -2.2, -1.4, -0.7, 0.1, 0.4, 0.6, 0.8, 1.1, 1.0, 1.2, 1.3]
])

# Välj en station för analys, t.ex. första stationen
x = np.arange(data.shape[1])  # Tid (index)
y = data[0]  # Temperatur för "Halmstad flygplats"

# Linjär regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Predikterade värden
y_pred = intercept + slope * x

# Konfidensintervall för regressionens lutning och intercept
n = len(y)
t_value = t.ppf(0.975, df=n-2)  # För 95% CI
conf_interval_slope = t_value * std_err
conf_interval_intercept = t_value * std_err * np.sqrt(np.mean(x**2))

# Visualisera regressionen och konfidensintervall
plt.figure(figsize=(10, 6))

# Plotta originaldata
plt.scatter(x, y, label="Originaldata", color="blue", alpha=0.7)

# Plotta regressionslinjen
plt.plot(x, y_pred, label=f"Regressionslinje: y = {intercept:.2f} + {slope:.2f}x", color="red")

# Plotta konfidensintervall
ci_upper = y_pred + conf_interval_slope
ci_lower = y_pred - conf_interval_slope
plt.fill_between(x, ci_lower, ci_upper, color='pink', alpha=0.3, label="95% Konfidensintervall")

# Grafikinställningar
plt.title("Linjär regression med 95% konfidensintervall")
plt.xlabel("Tid (index)")
plt.ylabel("Temperatur (°C)")
plt.legend()
plt.grid()
plt.show()

# Skriv ut parametrar
print(f"Lutning (slope): {slope:.2f} ± {conf_interval_slope:.2f}")
print(f"Intercept: {intercept:.2f} ± {conf_interval_intercept:.2f}")
