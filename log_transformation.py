"""
Log-Transform data
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.special import expit  # För exponentiering av modellen

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


# Temperaturdata för första stationen (Halmstad flygplats)
x = np.arange(data.shape[1])  # Tid (index)
y = data[0]  # Temperaturer

# Log-transformera data
y_log = np.log(y - np.min(y) + 1)  # Flytta datan för att undvika log(0)

# Linjär regression på transformerad data
slope_log, intercept_log, r_value_log, p_value_log, std_err_log = linregress(x, y_log)

# Återtransformera modellen till ursprunglig skala
y_log_pred = intercept_log + slope_log * x  # Prediktion i log-skala
y_exp_pred = np.exp(y_log_pred) - 1 + np.min(y)

# Linjär regression på ursprunglig data
slope, intercept, r_value, p_value, std_err = linregress(x, y)
y_pred = intercept + slope * x

# Visualisera
plt.figure(figsize=(12, 8))

# Originaldata
plt.scatter(x, y, label="Originaldata", color="blue", alpha=0.7)

# Linjär modell
plt.plot(x, y_pred, label=f"Linjär modell: y = {intercept:.2f} + {slope:.2f}x", color="red")

# Transformerad modell
plt.plot(x, y_exp_pred, label="Transformerad modell (log->exp)", color="green")

# Grafikinställningar
plt.title("Jämförelse av linjär och transformerad regression")
plt.xlabel("Tid (index)")
plt.ylabel("Temperatur (°C)")
plt.legend()
plt.grid()
plt.show()

# Skriv ut parametrar
print(f"Linjär modell: Lutning = {slope:.2f}, Intercept = {intercept:.2f}")
print(f"Transformerad modell: Lutning = {slope_log:.2f}, Intercept = {intercept_log:.2f}")
