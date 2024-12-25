## This import is inactivated due to inactivatet part for downloading of data
# import requests

# TO save json and download json
import json
# # to be abble create ne files and dir
# import os

# change timestamt to time
from datetime import datetime
import math
import pytz
## statisticks and figures
import requests
import scipy.stats as sci
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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

data_rows = []

# Create dictionary for three days data form each station in accending order
for param_id, parameter in PARAMS.items():
    three_d_station = {}
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
        three_d_station[name] = stat_set

        three_days[param_id] = three_d_station

# Convert the list of dictionaries into a pandas DataFrame objekt
df_three = pd.DataFrame(data_rows)
df_three.columns.str.replace(' ', '\n')
stations = df_three['station_name'].unique()
parameters = df_three['parameter'].unique()


# save to markdown file to be able sow in the presentation
utils.save_to_mdfile(df_three, 'dataframe.md', 'statistics')

# CHECK FOR MISSING POINTS
# Count NaN valu3es per station_name and parameter
nan_counts = df_three.groupby(['station_name', 'parameter'])['value'].apply(lambda x: x.isna().sum()).reset_index()

# Give name for columns
nan_counts.columns = ['station_name', 'parameter', 'Missing values']
utils.save_to_mdfile(nan_counts, "nan_counts.md", "statistics")

# descriptive statistics
descriptive_stats = df_three.groupby(['station_name', 'parameter'])['value'].describe()
utils.save_to_mdfile(descriptive_stats.round(2), "descriptive_stats.md", "statistics")

"""
Create figur showing frequensy destérsion of values for all stations and parameters
"""
stations = df_three['station_name'].unique()
parameters = df_three['parameter'].unique()

plt.figure(figsize=(8, 6)) # initiate figure
# Prepare the custom blue square legend handle
text = f"Blue color shows samples distribution"
blue_square = Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=8, label=text)

# Prepare the legend for the normal distribution
normal_dist_line = Line2D([0], [0], color='orange', lw=2, label="Normal Distribution")

normal_dist_added = False # variable to chose what norm dist line vill be shown in the legend
# Iterate through all stations and parameters
for i, station in enumerate(stations):
    for j, parameter in enumerate(parameters):
        # filter data for each station and parameter
        data = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]

        # Subplot indexering: 3 rows for 3 stations and 2 columns for 2 parameters
        plt.subplot(3, 2, i * len(parameters) + j + 1) 
        
        sns.histplot(data['value'], kde=True, bins=24, color="blue", edgecolor="black")
                
        # Calculate the mean and standard deviation
        mean = data['value'].mean()
        std_dev = data['value'].std()

        # Generate x values for normal distribution (range around the data's values)
        x = np.linspace(data['value'].min(), data['value'].max(), 100)
        
        # Calculate the normal distribution values (PDF)
        y = sci.norm.pdf(x, mean, std_dev)
        # Add normal distribution with te same parameters to the subplot
        plt.plot(x, y * len(data) * (x[1] - x[0]), color='orange')
        
        # add title and axes
        plt.title(f"{station}", fontsize=10)
         # Conditionally set the xlabel depending on the parameter
        if parameter == 'TEMPERATUR':
            plt.xlabel(f"{parameter.lower()} (°C)")
        else:
            plt.xlabel(f"{parameter.lower()} (%)")
        
        plt.ylabel("Frekvens")
# Create a global legend outside the subplots (top)
fig = plt.gcf()  # Get the current figure

fig.legend(handles=[blue_square, normal_dist_line], loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize='small')

plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Adjust top margin to make room for the legend

# Save and show the plot
plt.savefig("img/frekvenser/alla.png")
plt.show()
plt.close()

"""
BOX plots and Shapiro_Wilk test
"""
# Unique stations and parameters
stations = df_three['station_name'].unique()
parameters = df_three['parameter'].unique()

# Set up the figure
fig, axes = plt.subplots(2, 3, figsize=(12, 4 * 2))  # 2 rows, 3 columns

results = []
# Loop over stations and parameters
for i, parameter in enumerate(parameters):
    for j, station in enumerate(stations):
        data_filtered = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]
        stat, p_value = sci.shapiro(data_filtered['value'])
        results.append({
            'Station': station,
            'Parameter': parameter,
            'Shapiro-Wilk Statistic': round(stat, 5),
            'P-value': round(p_value, 5),
            'Normal Distribution (p > 0.05)': 'Yes' if p_value > 0.05 else 'No'
        })
        ax = axes[i, j]
        
        # the boxplot
        sns.boxplot(
            ax=ax,
            data=data_filtered,
            x='station_name',  # Same station on x-axis
            y='value',
            hue='station_name',
            palette=[COLORS[j]],  # Assign unique color for the station
            width=0.3,
            dodge=False
        )
        ax.set_title(f"{station} - {parameter}", fontsize=8)
        ax.set_ylabel(f"{'°C' if parameter == 'TEMPERATUR' else '%'}", fontsize=8)

        # Annotate p-value on the plot
        ax.text(
            0.9, 0.8,  # Position: center-top of the plot
            f"p={p_value:.5f}",
            transform=ax.transAxes,
            fontsize=10,
            ha='center',
            color='red' if p_value < 0.05 else 'black'
        )
plt.tight_layout()
plt.savefig('img/box_plot/all.png')
plt.show()
plt.close()
# Save the results of Shapiro-Wilk test
results_df = pd.DataFrame(results)
utils.save_to_mdfile(results_df, "shapiro_wilk.md", "statistics")

""""
Q_Q plottar
"""
fig, axes = plt.subplots(2, 3, figsize=(10, 3 * 2))
# Loopa through all stations and parameters
for i, station in enumerate(stations):
    for j, parameter in enumerate(parameters):
        # Filter for station and oarameter
        data = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]
        numeric_data = data['value'].dropna()
        # Create Q_Q plots
        ax = axes[j, i]
        sci.probplot(numeric_data, dist="norm", plot=ax)
        ax.set_ylabel(f"{'temperatur, °C' if parameter == 'TEMPERATUR' else 'humidity, %'}", fontsize=8)
        # Add titel
        ax.set_title(f"Q-Q plot: {station} - {parameter}", fontsize=8)
        ax.get_lines()[1].set_color('red')  # Give lene for the 
plt.tight_layout()
plt.savefig('img/q_q_plot/all.png')
plt.close()

""""
Q_Q plottar without outliers (ex for Umeå)
"""
# CHANGE DATA
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for i, value in enumerate(PARAMS.values()):
    ax = axes[i]
    # Filter data for the specific station and parameter
    data = df_three[(df_three['station_name'] == 'Halmstad flygplats') & (df_three['parameter'] == value[0])]

    # Sort the data by 'value' column to identify outliers
    sorted_data = data.sort_values(by='value')

    # Remove two highest and one lowest value
    filtered_data = sorted_data.iloc[4:]  # Removes the first .. (lowest) and last .. (highest) rows
    iloc = "4:"
    # Modify the station name to reflect the adjustment
    filtered_data = filtered_data.copy()  # By DataFrame object metod copy() by defoult make deep copy
    filtered_data['station_name'] = f"Halmstad {value[0]} minus outliers"
    stat, prob = sci.shapiro(filtered_data['value'])
    numeric_data = filtered_data['value'].dropna()

    sci.probplot(numeric_data, dist="norm", plot=ax)
    ax.set_ylabel(f"{'temperatur, °C' if value[0] == 'TEMPERATUR' else 'humidity, %'}", fontsize=8)
    axes[i].text(0.1, 0.9, 
                f"Shapiro_Wilk test: statistics={stat},\n probubility for normal distribution={prob}", 
                color="red", fontsize=5,
                transform=ax.transAxes, 
                verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.5))
    # Add titel
    ax.set_title(f"Q-Q plot: Halmstad - {value[0]}, removed {iloc}", fontsize=8)
    ax.get_lines()[1].set_color('red')  # Give line of teoretish quatnils color (red) 
plt.tight_layout()
plt.savefig('img/q_q_plot/Halmstad_min_outliers.png')
plt.show()
plt.close()

"""
CORRELATION MATRIX FOR 6 PARAMETERS
"""
# Combine data for all parameters, with station names included
combined_data = pd.DataFrame()
column_name1 = "TEMPERATUR_Umeå Flygplats"
column_name2 = "LUFTFUKTIGHET_Umeå Flygplats" 
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
plt.savefig('img/correlations/all_correlations.png')
plt.show()
plt.close()

# Pairplot 
f = sns.pairplot(combined_data, height=1.8, diag_kind='kde')
# Adjust font size and axis
for ax in f.axes.flatten():
    # Get current x and y axis labels
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlabel = re.sub(r'(?i)flygplats', '', xlabel).strip()  # Remove "flygplats" (case-insensitive)
    xlabel = xlabel.replace("TEMPERATUR_", "°C, TEMP_").strip()  # Replace "TEMPERATUR_" with "TEMP_"
    xlabel = xlabel.replace("LUFTFUKTIGHET_", "%, FUKT_").strip()  # Replace "LUFTFUKTIGHET_" with "FUKT_"
    
    ylabel = re.sub(r'(?i)flygplats', '', ylabel).strip()  # Remove "flygplats" (case-insensitive)
    ylabel = ylabel.replace("TEMPERATUR_", "°C, TEMP_").strip()  # Replace "TEMPERATUR_" with "TEMP_"
    ylabel = ylabel.replace("LUFTFUKTIGHET_", "%, FUKT_").strip()  # Replace "LUFTFUKTIGHET_" with "FUKT_"
    
    # Set the modified labels with font size
    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    
    ax.set_ylabel(ylabel.replace("LUFTFUKTIGHET_", "%, FUKT_").strip(), fontsize=6)
    
    # Set font size for tick labels
    ax.tick_params(axis='x', labelsize=5)  # X-axis tick labels
    ax.tick_params(axis='y', labelsize=5)  # Y-axis tick labels

plt.suptitle("Pairwise Relationships for Parameters and Stations", y=0.99, fontsize=16)  # Title for the plot
plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9) # Ajust spase between subplots
plt.savefig('img/regression/all_pairwise_relationships.png')
plt.show()
plt.close()

"""
Calculate regression for FUKT dependently on TEMP in UMEÅ
"""
# X (independent variable) and y (dependent variable)
X = combined_data[column_name1].values.reshape(-1, 1)  # Reshape for a single feature
y = combined_data[column_name2].values

# Initialize the model
model = LinearRegression()

# Fit data to the model
model.fit(X, y)

# Get the regression line parameters (slope and intercept)
slope = model.coef_[0] # =b
intercept = model.intercept_ # =a

# Print the regression equation
print(f"Regression Equation: y = {slope:.2f} * X + {intercept:.2f}")

# Plot the data and regression line
plt.scatter(X, y, color='blue', label='Data Points')  # Plot the data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot the regression line

# Customize the plot
plt.xlabel(column_name1)
plt.ylabel(column_name2)
plt.title(f"Linear regression model.\nPrediktion av relativt luftfuktighet på grund of temperatur i Umeå")

# label for the plot
sns.regplot(x=column_name1, y=column_name2, data=combined_data, scatter_kws={'s': 10}, line_kws={'color': 'red', 'label': f'Y = {slope:.2f}X + {intercept:.2f}'})
# get variables to chage legend
handles, labels = plt.gca().get_legend_handles_labels()

# Remove "Regression Line" from legend if it exists
handles = [handle for handle, label in zip(handles, labels) if label != 'Regression Line']
labels = [label for label in labels if label != 'Regression Line']

# Add the correct custom legend (keeping only data points and regression equation)
plt.legend(handles=handles, labels=labels, loc='best')# remove regression line from legend as it will be desplayed by sns.regplot
plt.savefig('img/regression/Umea_temp_fukt_relation.png')
plt.close()

"""
Regression for FUKT dependently on TEMP in UMEÅ
with train and test data
"""
# Get training ang testing datasets
fraktion = 0.5
train = combined_data.sample(frac=fraktion, random_state=1)
test = combined_data.drop(train.index)

# # Extract X (independent variable) and y (dependent variable) from the dataframe
X_train = train[column_name1].values.reshape(-1, 1)  # Reshape for a single feature
y_train = train[column_name2].values  # Dependent variable (y)
X_test = test[column_name1].values.reshape(-1, 1)  # Reshape for a single feature
y_test = test[column_name2].values  # Dependent variable (y)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# Calculate MSE
mse = np.mean((pred - y_test)**2)
linear_slope = model.coef_[0]
linear_intercept = model.intercept_

plt.figure(figsize=(10,6))
# Add linear regression parameters to the plot
plt.text(0.5, 0.95, f'Linear Model: y = {linear_slope:.2f}x + {linear_intercept:.2f}',
        ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')
# Visulisera prediktioner
plt.scatter(X_train, y_train, color="orange", label='Träningsdata', alpha=0.6)
plt.scatter(X_test, y_test, color="blue", label='Test data', alpha=0.6)
# Create the regression plot with a confidence interval (95%)
sns.regplot(x=column_name1, y=column_name2, data=combined_data, scatter=False, 
            line_kws={'color': 'red', 'label': f'Y = {linear_slope:.2f}X + {linear_intercept:.2f}'}, 
            ci=95)
# Add regression line from the model's predictions (for test data)
y_pred = model.predict(X_test)

plt.plot(X_test, y_pred, color='green', label='Test Data Prediction', linewidth=2)

plt.legend()
plt.title(f"Prediktioner av luftfuktighet på temperatur i Umeå\nMean squared error: {mse}" + 
        f"\nFraktion: {fraktion}")
plt.xlabel("Temperatur, °C")
plt.ylabel("Relativt Luftfuktighet, %")
plt.savefig(f'img/regression/regr_prediction_Umea_temp_luft_{fraktion}.png')
plt.close() 

# Calculate residuals of test data
residual = y_test - pred

# Calculate standarddeviation of residuals
std_residual = np.std(residual)

# Visualise residuals
plt.scatter(X_test, residual)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residualer av relativt luftfuktighet i Umeå")
plt.xlabel("temperatur i Umeå, °C")
plt.ylabel("Residualer")
# Place text at a specific location in data coordinates
plt.text(X_test[19], residual[19], f"Standard avvikelse för residualer: {std_residual:.2f}", 
        color="green", fontsize=8, ha='left', va='bottom')
plt.savefig('img/regression/residuals_temp_fukt_UME.png')
# plt.show()
plt.close()

# Show histogram av residualen för test data
plt.hist(residual, bins=24, edgecolor='white', alpha=1.0)
plt.title("Histogram av residualer av luftfuktighet i Umeå", fontsize=10)
plt.xlabel("Residuals")
plt.ylabel("Frekvens")
plt.savefig('img/regression/residuals_hist_temp_fukt_UME.png')
# plt.show()
plt.close() 

"""
MODIFATED DATA

TRANSFORMATION temperatur och fuktighet i Umeå
"""
# Log transformation
# I cannot use direct log-transformaton due to negative values of the tempture
X_combined = combined_data[column_name1].values

shift_value = abs(X_combined.min()) + 1e-5  # Make no zero values

X_train_log = np.log(X_train + shift_value)
X_test_log = np.log(X_test + shift_value)

# Sort the log-transformed values to identify outliers and track original indices
sorted_log = np.sort(X_test_log, axis=0)  # Sort along axis 0 (values)
sorted_log_indices = np.argsort(X_test_log, axis=0)  # Get the sorted indices

# Now filter the sorted values (e.g., remove the first and last elements)
filtered_sorted_log = sorted_log[1:]  # Remove the smallest and largest elements
filtered_sorted_indices = sorted_log_indices[1:]  # Corresponding indices after filtering

# Now, map the filtered data back to their original indices
X_test_log_filtered = np.copy(X_test_log)
y_test_filtered = np.copy(y_test)

# Set the outlier values (smallest and largest) to NaN (or any other replacement value)
X_test_log_filtered[sorted_log_indices[0]] = np.nan  # Remove the smallest value
X_test_log_filtered[sorted_log_indices[-1]] = np.nan  # Remove the largest value

# Remove corresponding values from y_test
y_test_filtered[sorted_log_indices[0]] = np.nan  # Remove corresponding y_test value
y_test_filtered[sorted_log_indices[-1]] = np.nan  # Remove corresponding y_test value
# Visualise log transformation
plt.figure(figsize=(12, 6))
# Show original data on subplot 1
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.legend()
plt.title("Original data")
plt.xlabel("Temperatur, °C")
plt.ylabel("Relativt luftfuktighet; %")

# Show log transformation on subplot 2
plt.subplot(1,2,2)
plt.scatter(X_train_log, y_train, label='Träningsdata')
plt.scatter(X_test_log_filtered, y_test_filtered, label='Test data')
plt.legend()
plt.title("Logaritmisk transformation")
plt.xlabel("Temperatur, log(value - min - 0.00001 )")
plt.ylabel("Relativt luftfuktighet; %")
plt.savefig('img/regression/original_and_log_data.png')
# plt.show()
plt.close()

# Create liniar regression with log tempture
log_model = LinearRegression()
log_model.fit(X_train_log, y_train)

# Make prediction with test data
pred_log = log_model.predict(X_test_log)

x_log = np.linspace(-0.2, 3.0, 100)
draw_exp_model = log_model.predict(x_log.reshape(-1, 1))

# Calculate MSE
mse_log = np.mean((pred_log - X_test_log)**2)
print("Mean squared error log transformerad:", mse_log)
# Transform predictions back to the original scale
pred_original = np.exp(pred_log) - shift_value

# Calculate MSE in the original domain
mse_original = np.mean((pred_original - X_test)**2)
print("Mean squared error log transformation:", mse_original)
print("Mean squared error linjär regression:", mse)

# Show prediktions
plt.scatter(X_train_log, y_train, label='Träningsdata')
plt.scatter(X_test_log_filtered, y_test_filtered, label='Test data')
plt.plot(x_log, draw_exp_model, label='Linjär regression, log transformerad i x', color='c', linewidth=3)
plt.legend()
plt.title("Prediktioner av relativt luftfuktighet (log transformerad)")
plt.xlabel("Temperatur [log( °C)]")
plt.ylabel("Relativt luftfuktighet, %")
plt.savefig('img/regression/prediction_log_data_filtered.png')
plt.show()
plt.close()

# Calculate residuals of test data
residual = y_test_filtered - pred_log
# this array contain nan becouse of previouse manipulations with data
# I want to remove nan elements:
data_without_nan = [x for x in residual if not math.isnan(x)]
std_residual = np.std(data_without_nan)

# Sheow residuals of the test data
plt.scatter(X_test, residual)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residualer av temperatur i Umeå")
plt.xlabel("temperatur i Umeå,  °C")
# Add the MSE text to the legend
mse_text = (f"Mean squared error liniar regression: {mse:.2f}\n"
            f"Mean squared error log transformation: {mse_log:.2f}")
plt.text(X_test[17], residual[17], mse_text, color="red", fontsize=8)
plt.ylabel("Residual")
plt.savefig('img/regression/residuals_filtrerad_LOGtemp_fukt_UME.png')
plt.show()
plt.close()

# Show histogram of the residuals of the test data
plt.hist(residual, bins=10)
plt.title("Histogram av residualer av log_temperatur prediction av luftfuktighet i Umeå", fontsize=10)
plt.xlabel("Residual")
plt.ylabel("Frekvens")
plt.savefig('img/regression/residuals_filtrerad_hist_LOGtemp_fukt_UME.png')
plt.show()
plt.close() 

# Transform back model
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(np.exp(x_log) - shift_value, draw_exp_model, label='Linjär regression, exponentiell i x', color='c', linewidth=3)
plt.legend()
plt.title("Prediktioner av relativt luftfuktighet exponentiell modell")
plt.xlabel("Temperatur, °C")
plt.ylabel("Relativt luftfuktighet, %")
plt.savefig('img/regression/transform_back.png')
#plt.show()
plt.close()

# Regression with log relativt humidity
# Relative humidity is always pozitiv
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# create liniar regression pf the log data
log_y_model = LinearRegression()
log_y_model.fit(X_train, y_train_log)

# Make prediction of the test data
pred_log_y = log_y_model.predict(X_train)

# Calculate MSE
mse_log_y = np.mean((pred_log_y - y_test_log)**2)
print("Mean squared error log transformerad y:", mse_log_y)
print("Mean squared error log transformerad x:", mse_log)
print("Mean squared error linjär regression:", mse)

# Showlisera prediktioner
x = np.linspace(-21, -0.5, 100)
y_log = log_y_model.predict(x.reshape(-1, 1))

plt.scatter(X_train, y_train_log, label='Träningsdata')
plt.scatter(X_test, y_test_log, label='Test data')
plt.plot(x, y_log, label='Linjär regression log domän', color='g', linewidth=3)
plt.legend()
plt.title("Prediktioner av relativt luftfuktighet (log transformerad y, %)")
plt.xlabel("Temperatyr")
plt.ylabel("Relativt luftfuktighet [log %]")
plt.savefig('img/regression/log_transform_FUKT_Umeå.png')
plt.show()
plt.close()

# Calculate residualer
residual_log_y = y_test - np.exp(pred_log_y)

# MSE
mse_exp_y = np.mean(residual_log_y**2)
print(f"MSE av luftfuktighet (exponentiell modell i y): {mse_exp_y}")
print(f"Mse av luftfuktighet (log transformerad x): {mse_log}")
print(f"Mse av luftfuktighet (linjär regression): {mse}")

# Visualise exponential modell on y
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(x, np.exp(y_log), label='Linjär regression exponentiell i y', color='g', linewidth=3)
plt.legend()
plt.title("Prediktioner av luftfuktighet (exponentiell modell i y)")
plt.xlabel("Relativt lutfuktighet %")
plt.ylabel("Temperatur, °C")
plt.savefig('img/regression/Y_LOG_transform_model_Umeå.png')
plt.show()
plt.close()

#ALLA MODELLER PÅ EN PLOT
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(x, np.exp(y_log), label='Linjär regression exponentiell i y', color='orange', linewidth=3)
plt.plot(np.exp(x_log) - shift_value, draw_exp_model, label='Linjär regression, exponentiell i x', color='red', linewidth=3)
plt.plot(X_test, y_pred, color='green', label='Test Data Prediction', linewidth=2)
plt.title("Prediktioner av luftfuktighet av alla modeller")
plt.ylabel("Relativt lutfuktighet %")
plt.xlabel("Temperatur, °C")
plt.legend(loc='best', frameon=True, fontsize=10, title="Modeller och data", title_fontsize=10)
plt.tight_layout()
plt.savefig('img/regression/alla_modeller_Umeå.png')
plt.show()
