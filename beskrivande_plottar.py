"""
this file shows descriptions plots
"""
import os
import scipy.stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import get_dynam_data.prepere_data as p_d

data = p_d.extract_for_statistics()
print(data)
# Kombinera alla temperaturer
all_temperatures = [temp for temps in data.values() for temp in temps]
#print(all_temperatures)
# Skapa histogrammet
plt.figure(figsize=(10, 6))
sns.histplot(all_temperatures, kde=False, bins=15, color="blue", label="Temperaturdata", stat="density")

# Lägg till en normalfördelning
mean = np.mean(all_temperatures)
std = np.std(all_temperatures)
x = np.linspace(min(all_temperatures), max(all_temperatures), 100)
plt.plot(x, norm.pdf(x, mean, std), label="Normalfördelning", color="red", linewidth=2)

plt.title("Histogram av temperaturer jämfört med normalfördelning")
plt.xlabel("Temperatur (°C)")
plt.ylabel("Densitet")
plt.legend()
plt.show()

# Skapa ett boxplot för att jämföra stationerna
plt.figure(figsize=(10, 6))
sns.boxplot(data=list(data.values()), notch=True)

# Lägg till etiketter för stationerna
plt.xticks(ticks=range(len(data)), labels=data.keys(), rotation=15)
plt.title("Lådagram för temperaturer vid olika stationer")
plt.ylabel("Temperatur (°C)")
plt.show()
