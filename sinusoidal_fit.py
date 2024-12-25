import json

from scipy.optimize import leastsq, curve_fit

from datetime import datetime
import math
import pytz
## statisticks and figures
import requests
import scipy.stats as sci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import re  # Import regular expression module
from sklearn.metrics import r2_score, mean_squared_error
## own functions
import utils
# variables
STATIONS = {'Halmstad flygplats': 62410, 'Uppsala Flygplats': 97530, 'Umeå Flygplats': 140480}
COLORS = ["red"]
# number of columns each dataframe
NUM_COLUMNS = len(STATIONS)
# Directory to save the data files and statistics
OUTPUT_DIR = {"data":"smhi_data_temp_fukt", "img":"img", "statistics":"statistics"}
COLORS = ["orange", "yellow", "green"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "CustomCmap", COLORS, N=256
)
# parameters to download (parameter_id:parameter_name)
PARAMS = {1:["TEMPERATUR", "°C"], 6:["LUFTFUKTIGHET", "%"]}
# period to request. Available periods: latest-hour, latest-day, latest-months or corrected-archive
PERIOD = "latest-months"

# This part i inactivated becouse i work with downloaded data
"""# Dowloads data from three stations and for two parameters
for key in PARAMS.keys():
    for station, id in STATIONS.items():
        data_url = f'https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{key}/station/{id}/period/{PERIOD}/data.json'
        response = requests.get(data_url)
        response.raise_for_status()  # Check if the request succeeded
        
        result = json.loads(response.content)
        save_path = f'{OUTPUT_DIR["data"]}/{id}_{key}.json'
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)"""

# Extract requaired period (tree days) from downloaded data
mesured_points = 72 # how mach n will be in the data
#all_data = {}
three_days = {}
all_days = {}

data_rows = []

# Create dictionary for three days data form each station in accending order
for param_id, parameter in PARAMS.items():
    three_d_station = {}
    all_d_station = {}
    for name, station_id in STATIONS.items():
        file_path = OUTPUT_DIR["data"] + '/' + f'{station_id}_{param_id}.json'
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Extract the "value" list and sort it by timestamp
            sorted_data = sorted(
                data.get("value", []),
                key=lambda x: datetime.fromtimestamp(x["date"] / 1000, tz=pytz.timezone("Europe/Stockholm"))
            )
            # Get the last N points
            last_points = sorted_data[- mesured_points:]
            """
            change it to pivot tabel
            """
        """the arrays' item are dict with keys: date, value and quality. 
        I want remove quality but replace value to nympy.nan if quality is not G or Y
        """
        stat_set = {}
        for item in last_points:
            new_value = float(item['value']) if item['quality'] in ['G', 'Y'] else np.nan
            stat_set[item['date']] = new_value  # Add date-value pair to value_set
            time = datetime.fromtimestamp(item['date'] / 1000, tz=pytz.timezone("Europe/Stockholm"))
            value = float(item['value']) if item['quality'] in ['G', 'Y'] else np.nan

            data_rows.append({
                'time': time,
                'station_name': name,
                'parameter': PARAMS[param_id][0],
                'value': value
            })
        all_d_station[name] = stat_set

        all_days[param_id] = three_d_station

# Convert the list of dictionaries into a pandas DataFrame
df_all = pd.DataFrame(data_rows)

df_all.columns.str.replace(' ', '\n')
# stations = df_all['station_name'].unique()
# parameters = df_all['parameter'].unique()

# Get table with only TEMPERATUR parameter
df_cleaned_temp = df_all[df_all['parameter'] == 'TEMPERATUR']

# Create 'hour72' and handle duplicate timestamps by averaging values
data_temp = df_cleaned_temp.copy()

# Calculate 'hour72'
data_temp['hour72'] = ((data_temp['time'] - data_temp['time'].min()).dt.total_seconds() // 3600).astype(int) % 72

# Pivot the table with 'time' as the index
pivoted_temp = data_temp.pivot_table(index='hour72', columns='station_name', values='value')

# Reset index to bring 'time' back as a column
pivoted_temp = pivoted_temp.reset_index()

# Train/test split (50% for training and 50% for testing)
train_temp = pivoted_temp.sample(frac=0.5, random_state=1)

# # I will se if I can have regression as sinusoid y=A⋅sin(B⋅x+C)+D

test_temp = pivoted_temp.drop(train_temp.index)
X = pivoted_temp['hour72'].values
y = pivoted_temp["Umeå Flygplats"].values
X_train = train_temp['hour72'].values
y_train = train_temp["Umeå Flygplats"].values

X_test = test_temp['hour72'].values
y_test = test_temp["Umeå Flygplats"].values
# LINIAR REGRESSION
liniar_model = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
linear_pred = liniar_model.predict(X_test.reshape(-1, 1))
linear_mse = np.mean((linear_pred - y_test)**2)
print("Mean squared error for liniar regression:", linear_mse)

# Define sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D
mean_temp = y.mean()
std_temp = y.std()
initial_guess = [std_temp, 24, 3, mean_temp]  # Amplitude, Frequency, Phase Shift, Vertical Shift
params, params_covariance = curve_fit(sinusoidal, X_train, y_train, p0=initial_guess)
sinusoid_pred = sinusoidal(X_test, *params)
print(params)

# Calculate Sinusoidal Modl Mean Squared Error (MSE)
sinusoid_mse = np.mean((sinusoid_pred - y_test) ** 2)
print("Mean squared error for sinusoidal regression:", sinusoid_mse)
# Plot the results
plt.figure(figsize=(10, 6))
plt.xlim(0, 71)
plt.scatter(X_train, y_train, label='Trainings data', color='orange')
plt.scatter(X_test, y_test, label='Test data', color='blue')
plt.plot(X_test, linear_pred, label='Linear Regression', color='red')
plt.plot(X_test, sinusoid_pred, label='Sinusoidal Fit', color='green')
 # diapason(from 1 to 23)
plt.xlabel('tiime, h')
plt.ylabel('temperatur, °C')
plt.title('Linear vs Sinusoidal Regression')
# Display the parameters of the regression models in the plot
# Linear regression parameters
linear_slope = liniar_model.coef_[0]
linear_intercept = liniar_model.intercept_

# Add linear regression parameters to the plot
plt.text(0.5, 0.95, f'Linear Model: y = {linear_slope:.2f}x + {linear_intercept:.2f}',
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')

# Sinusoidal regression parameters (A, B, C, D)
A, B, C, D = params
# Add sinusoidal parameters to the plot
plt.text(0.6, 0.9, f'Sinusoidal Model: y = {A:.2f}sin({B:.2f}x + {C:.2f}) + {D:.2f}',
         ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='green')


# Print the results
text = f"MSE for Linear Regression: {linear_mse:.4f}\n" + f"MSE for Sinusoidal Fit: {sinusoid_mse:.4f}"
# Add MSE values as a separate legend entry
plt.plot([], [], ' ', label=text)
# Show the legend and grid
plt.legend()
path = f'img/regression/time_temp_Sinusoid_HALM.png'
plt.savefig(path)
plt.grid(True)

plt.show()
plt.close()
