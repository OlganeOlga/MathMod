## This import is inactivated due to inactivatet part for downloading of data
# import requests

# TO save json and download json
import json
# # to be abble create ne files and dir
# import os

# change timestamt to time
from datetime import datetime
import pytz
## statisticks and figures
import scipy.stats as sci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# for non liniar regression
from scipy.optimize import curve_fit

## own functions
import utils
# variables
STATIONS = {'Halmstad flygplats': 62410, 'Uppsala Flygplats': 97530, 'Umeå Flygplats': 140480}
COLORS = ["red"]
# number of columns each dataframe
NUM_COLUMNS = len(STATIONS)
# Directory to save the data files and statistics
OUTPUT_DIR = {"data":"smhi_data_temp_fukt", "img":"img", "statistics":"statistics"}
#os.makedirs(OUTPUT_DIR["data"], exist_ok=True)
COLORS = ["orange", "yellow", "green"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "CustomCmap", COLORS, N=256
)
# parameters to download (parameter_id:parameter_name)
PARAMS = {1:["TEMPERATUR", "°C"], 6:["LUFTFUKTIGHET", "%"]}
# period to request. Available periods: latest-hour, latest-day, latest-months or corrected-archive
PERIOD = "latest-months"

# This part i inactivated becouse i work with downloaded data
"""# Dowloads data from tree station and for two parameters
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

data_rows = []

# Create dictionary for three days data form each station in accending order
for param_id, parameter in PARAMS.items():
    #station_data = {}
    three_d_station = {}
    for name, station_id in STATIONS.items():
        file_path = OUTPUT_DIR["data"] + '/' + f'{station_id}_{param_id}.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
            #station_data[name] = data
            # Extract the "value" list and sort it by timestamp
            sorted_data = sorted(
                data.get("value", []),
                key=lambda x: datetime.fromtimestamp(x["date"] / 1000, tz=pytz.timezone("Europe/Stockholm"))
            )
            # Get the last N points
            last_points = sorted_data[-mesured_points:]
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

            # Append each row with the timestamp, station, parameter_id, and value
            data_rows.append({
                'time': time,
                'station_name': name,
                'parameter': PARAMS[param_id][0],
                'value': value
            })
        three_d_station[name] = stat_set
        
        three_days[param_id] = three_d_station
# chage dataset for each parameter into pandas DateFrame


# Convert the list of rows into a pandas DataFrame
df_three = pd.DataFrame(data_rows)

df_three.columns.str.replace(' ', '\n')
#print(df_three)

# Pivot the data so each station gets its own column, and timestamps are aligned
df_three_pivoted = df_three.pivot_table(index="time", columns=["station_name", "parameter"], values="value", aggfunc="mean")

# flatten pivottabel
df_md = df_three_pivoted.to_markdown()

utils.save_to_mdfile(df_md, 'df_pivoted.md', OUTPUT_DIR['statistics'])
# Display the DataFrame
stats = df_three_pivoted.describe()
# Flatten the MultiIndex columns
stats.columns = [' '.join(col).strip() for col in stats.columns.values]

# Display the flattened DataFrame
#print(stats.round(2))
md_tabel = utils.change_to_markdown(stats.round(2), None)
utils.save_to_mdfile(md_tabel, 'describe_stat_all.md', OUTPUT_DIR['statistics']) 

# Summary of missing data
missing_summary = df_three_pivoted.isna().sum()
# print(missing_summary.to_markdown())

"""
CORRELATION MATRIX FOR 6 PARAMETERS
"""
# Combine data for all parameters, with station names included
combined_data = pd.DataFrame()

for param_key, dataset in three_days.items():
    param_name = PARAMS[param_key][0]  # Parameter name (e.g., Temperature, Humidity)
    df = pd.DataFrame(dataset)
    df.columns = [f"{param_name}_{station}" for station in df.columns]  # Add station to column names
    combined_data = pd.concat([combined_data, df], axis=1)

# Calculate the correlation matrix
correlation_matrix = combined_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    correlation_matrix, 
    annot=True,  # Avoid cluttering with too many annotations
    cmap=CUSTOM_CMAP, 
    cbar=True
)
# Adjust font sizes for station names
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.title("Correlation Matrix for All Parameters and Stations", fontsize=14)
plt.tight_layout()
#plt.savefig('img/correlations/all_parameters_stations_correlation.png')
plt.show()
exit()
"""
regression for TEMPERATUR predicted by the hour in Halmstad flugplats
"""
# prepere date to regression investigation
df_three_temp = df_three[df_three['parameter'] == 'TEMPERATUR']
df_three_temp = df_three_temp.pivot_table(index='time', columns='station_name', values='value')

# Ensure correct data indexing for sampling
df_three_temp = df_three_temp.reset_index()

# Add 'hour' feature from the time index
df_three_temp['hour'] = df_three_temp['time'].dt.hour

# Train/test split (50% for training and 50% for testing)
train_temp = df_three_temp.sample(frac=0.5, random_state=1)
# # I will se if I can have regression as sinusoid y=A⋅sin(B⋅x+C)+D

test_temp = df_three_temp.drop(train_temp.index)

X_train = train_temp['hour'].values
y_train = train_temp["Halmstad flygplats"].values

X_test = test_temp['hour'].values
y_test = test_temp["Halmstad flygplats"].values
# LINIAR REGRESSION
liniar_model = LinearRegression().fit(X_train.reshape(-1, 1), y_train)
linear_pred = liniar_model.predict(X_test.reshape(-1, 1))
linear_mse = np.mean((linear_pred - y_test)**2)
print("Mean squared error for liniar regression:", linear_mse)

# Define sinusoidal function
def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

initial_guess = [10, 2 * np.pi / 24, 0, np.mean(y_train)]  # Amplitude, Frequency, Phase Shift, Vertical Shift
params, params_covariance = curve_fit(sinusoidal, X_train, y_train, p0=initial_guess)
sinusoid_pred = sinusoidal(X_test, *params)
# Calculate Sinusoidal Model Mean Squared Error (MSE)
sinusoid_mse = np.mean((sinusoid_pred - y_test) ** 2)
print("Mean squared error for sinusoidal regression:", sinusoid_mse)
# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Data', color='blue')
plt.plot(X_test, linear_pred, label='Linear Regression', color='red')
plt.plot(X_test, sinusoid_pred, label='Sinusoidal Fit', color='green')
plt.xlabel('Hour of Day')
plt.ylabel('Parameter')
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
print(f"Residual Sum of Squares for Linear Regression: {linear_mse:.2f}")
print(f"Residual Sum of Squares for Sinusoidal Fit: {sinusoid_mse:.2f}")

# Show the legend and grid
plt.legend()
path = f'img/regression/time_temp_Sinusoid_HALM.png'
plt.savefig(path)
plt.grid(True)

plt.show()
plt.close()
# Residuals for Linear Model
residuals_linear = y_test - linear_pred

# Residuals for Sinusoidal Model
residuals_sinusoid = y_test - sinusoid_pred
# Plot residuals distribution for linear regression
plt.figure(figsize=(12, 6))

# Linear regression residuals
plt.subplot(1, 2, 1)
plt.hist(residuals_linear, bins=30, color='red', edgecolor='black')
plt.title('Residuals Distribution for Linear Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Sinusoidal regression residuals
plt.subplot(1, 2, 2)
plt.hist(residuals_sinusoid, bins=30, color='green', edgecolor='black')
plt.title('Residuals Distribution for Sinusoidal Fit')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
plt.close()

# Shapiro-Wilk test for linear regression residuals
stat_linear, p_value_linear = sci.shapiro(residuals_linear)

# Shapiro-Wilk test for sinusoidal regression residuals
stat_sinusoid, p_value_sinusoid = sci.shapiro(residuals_sinusoid)

# Output the results
print(f"Linear Regression Residuals - Shapiro-Wilk Test: Stat={stat_linear:.3f}, p-value={p_value_linear:.3f}")
print(f"Sinusoidal Regression Residuals - Shapiro-Wilk Test: Stat={stat_sinusoid:.3f}, p-value={p_value_sinusoid:.3f}")

# Interpretation of p-values
if p_value_linear > 0.05:
    print("Linear regression residuals are normally distributed (p-value > 0.05).")
else:
    print("Linear regression residuals are NOT normally distributed (p-value <= 0.05).")

if p_value_sinusoid > 0.05:
    print("Sinusoidal regression residuals are normally distributed (p-value > 0.05).")
else:
    print("Sinusoidal regression residuals are NOT normally distributed (p-value <= 0.05).")

# Compare performance
from sklearn.metrics import mean_squared_error, r2_score

linear_mse = mean_squared_error(y_test, linear_pred)
sinusoid_mse = mean_squared_error(y_test, sinusoid_pred)

linear_r2 = r2_score(y_test, linear_pred)
sinusoid_r2 = r2_score(y_test, sinusoid_pred)

print(f"Linear Regression MSE: {linear_mse}, R²: {linear_r2}")
print(f"Sinusoidal Fit MSE: {sinusoid_mse}, R²: {sinusoid_r2}")

# Display sinusoidal parameters
print(f"Sinusoidal Parameters: A={params[0]:.2f}, B={params[1]:.2f}, C={params[2]:.2f}, D={params[3]:.2f}")


# sns.lmplot(x="Halmstad flygplats", y="Umeå Flygplats", data=df_pivoted_temp)
# plt.show()
train = df_three_temp.sample(frac=0.5, random_state=1)
test = df_three_temp.drop(train.index)
model = LinearRegression().fit(train[['Halmstad flygplats']], train['Umeå Flygplats'])
pred = model.predict(test[['Halmstad flygplats']])
# Räkna ut MSE
mse = np.mean((pred - test['Umeå Flygplats'])**2)
#print("Mean squared error:", mse)
# Visalisera prediktioner
plt.scatter(train['Halmstad flygplats'], train['Umeå Flygplats'], label='Träningsdata')
plt.scatter(test['Halmstad flygplats'], test['Umeå Flygplats'], label='Test data')
plt.plot(test['Halmstad flygplats'], pred, label='Linjär regression', color='g', linewidth=3)
plt.legend()
plt.title("Prediktioner av temperatur i Umeå på basis av temperatur i Halmstad")
plt.xlabel("Kronbladslängd")
plt.ylabel("Kronbladsbredd")
path = f'img/regression/regrTEMP_HALM_UME.png'
plt.savefig(path)
plt.close() 
# plt.show()


# Beräkna residualen för test data
residual = test['Umeå Flygplats'] - pred

# Beräkna standardavvikelsen för residualen
std_residual = np.std(residual)
#print(f"Standardavvikelsen för residualen: {std_residual:.2f}")

# Visualisera residualen för test data
plt.scatter(test['Halmstad flygplats'], residual)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residualer av temperatur i Umeå")
plt.xlabel("temperatur i Halmstad")
plt.ylabel("Residual")
path = f'img/regression/residuals_TEMP_HALM_UME.png'
plt.savefig(path)
# plt.show()
plt.close()

# Visa histogram av residualen för test data
plt.hist(residual, bins=10)
plt.title("Histogram av residualer av temperatur i Umeå")
plt.xlabel("Residual")
plt.ylabel("Frekvens")
# plt.show()
path = f'img/regression/residuals_hist_TEMP_HALM_UME.png'
plt.savefig(path)
plt.close() 


"""
Regression som kan prediktera lyftfuktighet på grund av temperatur
"""
#get data:
df_HALM = df_three[df_three['station_name'] == 'Halmstad flygplats']
df_pivoted_HALM = df_HALM.pivot_table(index='time', columns='parameter', values='value')
train = df_pivoted_HALM.sample(frac=0.5, random_state=1)
test = df_pivoted_HALM.drop(train.index)
model = LinearRegression().fit(train[['TEMPERATUR']], train['LUFTFUKTIGHET'])
pred = model.predict(test[['TEMPERATUR']])
# Räkna ut MSE
mse = np.mean((pred - test['LUFTFUKTIGHET'])**2)
#print("Mean squared error:", mse)
# Visalisera prediktioner
plt.scatter(train['TEMPERATUR'], train['LUFTFUKTIGHET'], label='Träningsdata')
plt.scatter(test['TEMPERATUR'], test['LUFTFUKTIGHET'], label='Test data')
plt.plot(test['TEMPERATUR'], pred, label='Linjär regression', color='g', linewidth=3)
plt.legend()
plt.title("Prediktioner av luftfuktighet på basis av temperatur i Halmstad")
plt.xlabel("Kronbladslängd")
plt.ylabel("Kronbladsbredd")
# path = f'img/regression/regrTEMP_HALM_UME.png'
# plt.savefig(path)
# plt.close() 
#plt.show()


# Beräkna residualen för test data
residual = test['LUFTFUKTIGHET'] - pred

# Beräkna standardavvikelsen för residualen
std_residual = np.std(residual)
#print(f"Standardavvikelsen för residualen: {std_residual:.2f}")

# Visualisera residualen för test data
plt.scatter(test['TEMPERATUR'], residual)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residualer av temperatur i Umeå")
plt.xlabel("temperatur i Halmstad")
plt.ylabel("Residual")
# path = f'img/regression/residuals_TEMP_HALM_UME.png'
# plt.savefig(path)
#plt.show()
plt.close()

# Visa histogram av residualen för test data
plt.hist(residual, bins=10)
plt.title("Histogram av residualer av temperatur i Umeå")
plt.xlabel("Residual")
plt.ylabel("Frekvens")
#plt.show()

# for column in df.loc[:, df.columns != 'time']:
#     print(len(df_three_pivoted[column]))
# path = f'img/regression/residuals_hist_TEMP_HALM_UME.png'
# plt.savefig(path)
# plt.close() 

# print(y1)
# y2 = df_pivoted_temp.iloc[:, 2].values
# print(len(y2))
# sns.lmplot(x='petallength', y='petalwidth', data=df_iris)
# # Split data
# X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)

# # Initialize and train model
# model = LinearRegression()
# model.fit(X_train, y1_train)

# # Make predictions
# y1_pred = model.predict(X_test)

# # Evaluate performance
# print("Mean Squared Error:", mean_squared_error(y1_test, y1_pred))
# print("R^2 Score:", r2_score(y1_test, y1_pred))
# plt.scatter(y1_test, y1_pred, alpha=0.5)
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs. Predicted")
# # Plot the regression line
# plt.plot(y1_test, y1_pred, color='red', linewidth=2)

# plt.show()
# # liniar regression:
# # Visalize pairwise relationships in a dataset filtered on class


# # # Select two stations for analysis
# # station1 = "TEMPERATUR"
# # station2 = "Uppsala Flygplats"

# # # Drop rows with missing values
# # analysis_df = pivot_df[[station1, station2]].dropna()

# # # Regression
# # X = analysis_df[station1].values.reshape(-1, 1)
# # y = analysis_df[station2].values

# # # Fit regression model
# # model = LinearRegression()
# # model.fit(X, y)

# # # Get regression line
# # y_pred = model.predict(X)

# # # Calculate correlation
# # correlation, _ = pearsonr(analysis_df[station1], analysis_df[station2])

# # # Print results
# # print(f"Regression equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
# # print(f"Pearson correlation: {correlation:.2f}")

# # # Plot the results
# # plt.scatter(X, y, color="blue", label="Actual data")
# # plt.plot(X, y_pred, color="red", label="Regression line")
# # plt.xlabel(station1)
# # plt.ylabel(station2)
# # plt.title(f"Temperature Regression: {station1} vs {station2}")
# # plt.legend()
# # plt.show()