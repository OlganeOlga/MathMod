## This import is inactivated due to inactivatet part for downloading of data
# import requests

# TO save json and download json
import json
# create new files and dir
import os
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
import statsmodels.api as sm
import re  # Import regular expression module
from sklearn.metrics import r2_score, mean_squared_error
## own functions
import utils
# variables
STATIONS = {'Halmstad flygplats': 62410, 'Uppsala Flygplats': 97530, 'Umeå Flygplats': 140480}
# colors to use in the plots
COLORS = ["orange", "yellow", "green"]
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
        save_path = f'data/{id}_{key}.json'
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)"""

# Extract the required period (three days) from downloaded data
measured_points = 72  # Number of points to include
three_days = {}
data_rows = []

# # Create dictionary for three days data form each station in accending order
# for param_id, parameter in PARAMS.items():
#     three_d_station = {}
#     for name, station_id in STATIONS.items():
#         file_path = 'data/' + f'{station_id}_{param_id}.json'
#         with open(file_path, 'r') as file:
#             data = json.load(file)

#             # Extract the "value" list and sort it by timestamp
#             sorted_data = sorted(
#                 data.get("value", []),
#                 key=lambda x: datetime.fromtimestamp(x["date"] / 1000, tz=pytz.timezone("Europe/Stockholm"))
#             )
#             # Get the last N points
#             last_points = sorted_data[- mesured_points:]
#             """
#             change it to pivot tabel
#             """
#         """the arrays' item are dict with keys: date, value and quality. 
#         I want remove quality but replace value to nympy.nan if quality is not G or Y
#         """
#         stat_set = {}
#         for item in last_points:
#             new_value = float(item['value']) if item['quality'] in ['G', 'Y'] else np.nan
#             stat_set[item['date']] = new_value  # Add date-value pair to value_set
#             time = datetime.fromtimestamp(item['date'] / 1000, tz=pytz.timezone("Europe/Stockholm"))
#             value = float(item['value']) if item['quality'] in ['G', 'Y'] else np.nan

#             data_rows.append({
#                 'time': time,
#                 'station_name': name,
#                 'parameter': PARAMS[param_id][0],
#                 'value': value
#             })
#         three_d_station[name] = stat_set

#         three_days[param_id] = three_d_station


# Process data for each parameter and station
for param_id, parameter in PARAMS.items():
    three_d_station = {}
    for name, station_id in STATIONS.items():
        file_path = f"data/{station_id}_{param_id}.json"
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Sort data by timestamp and select the last N points
        sorted_data = sorted(
            data.get("value", []),
            key=lambda x: datetime.fromtimestamp(x["date"] / 1000, tz=pytz.timezone("Europe/Stockholm"))
        )[-measured_points:]

        # Prepare station data and append rows for further processing
        stat_set = {}
        for item in sorted_data:
            new_value = float(item['value']) if item['quality'] in ['G', 'Y'] else np.nan
            stat_set[item['date']] = new_value
            data_rows.append({
                'time': datetime.fromtimestamp(item['date'] / 1000, tz=pytz.timezone("Europe/Stockholm")),
                'station_name': name,
                'parameter': PARAMS[param_id][0],
                'value': new_value
            })
        three_d_station[name] = stat_set
    three_days[param_id] = three_d_station

# Convert the list of dictionaries into a pandas DataFrame object
df_three = pd.DataFrame(data_rows)
df_three.columns.str.replace(' ', '\n')

# get stations and parameters from the DataFrame object
stations = df_three['station_name'].unique()
parameters = df_three['parameter'].unique()

# save to markdown file to be able sow in the presentation
utils.save_to_mdfile(df_three, 'dataframe.md', 'statistics')
def skip():
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
    BOX plots and Shapiro_Wilk test
    """

    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 4 * 2))  # 2 rows, 3 columns

    # Array t
    ShapiroW = []
    # Loop over stations and parameters
    for i, parameter in enumerate(parameters):
        for j, station in enumerate(stations):
            data_filtered = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]
            
            # Perform Shwpiro-Wilk normality test
            stat, p_value = sci.shapiro(data_filtered['value'])
            ShapiroW.append({
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
                hue = 'station_name',
                palette=[COLORS[j]],  # Assign unique color for the station
                width=0.3,
                dodge=False
            )
            #ax.set_title(f"{station} - {parameter}", fontsize=8)
            ax.set_ylabel(f"{parameter}, {'°C' if parameter == 'TEMPERATUR' else '%'}", fontsize=8)
            # Remove the x-axis label
            ax.set_xlabel("")

            # Annotate p-value on the plot
            ax.text(
                0.8, 0.13,  # Position: center-top of the plot
                f"p={p_value:.5f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='center',
                color='red' if p_value < 0.05 else 'black'
            )
    plt.suptitle("Databeskrivning med ladogrammar", fontsize=16)

    # Adjust layout to make space for the title
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 

    plt.savefig('img/box_plot/all.png')
    #plt.show()
    plt.close()
    # Save the results of Shapiro-Wilk test in a file
    ShapiroW_df = pd.DataFrame(ShapiroW)
    utils.save_to_mdfile(ShapiroW_df, "shapiro_wilk.md", "statistics")

    """
    Create figur showing frequensy despersion of values for all stations and parameters
    """

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
    #plt.show()
    plt.close()

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
    # Get data with removed titestips for the lowers temperatur
    name = 'Uppsala Flygplats'
    # Filter data for the specific station
    station_data = df_three[df_three['station_name'] == name]

    # Number of lowest temperature data points to remove
    to_remove = 4
    changed_by ='TEMPERATUR'
    # Find and remove the rows with the lowest temperature values
    param_data = station_data[station_data['parameter'] == changed_by]
    lowest_param = param_data.nsmallest(to_remove, 'value')  # Rows with the lowest parparameter values
    # remove even the row with the highest temprature value
    all_param = lowest_param.nlargest(1, 'value')
    #all_param = lowest_param
    all_param_timestamps = lowest_param['time'].tolist()  # Extract the timestamps as a list

    # Filter out rows with the lowest parameter values timestamps across all parameters
    filtered_data = station_data[~station_data['time'].isin(all_param_timestamps)]  # Use .isin() for filtering

    # Plot the filtered data
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i, value in enumerate(PARAMS.values()):
        ax = axes[i]
        # Filter the parameter-specific data
        param_data = filtered_data[filtered_data['parameter'] == value[0]]
        stat, prob = sci.shapiro(param_data['value'])
        numeric_data = param_data['value'].dropna()

        # Generate Q-Q plot
        sci.probplot(numeric_data, dist="norm", plot=ax)
        ax.set_ylabel(f"{'temperatur, °C' if value[0] == 'TEMPERATUR' else 'humidity, %'}", fontsize=8)
        ax.set_xlabel("teoretiska quantiler")
        axes[i].text(
            0.1, 0.9, 
            f"Shapiro_Wilk test: statistik={stat:.2f},\nsannolikhet för normalspridning={prob:.2f}", 
            color="red", fontsize=5,
            transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0.5)
        )
        # Clear the title
        ax.set_title(value[0].lower())
        # Add line
        ax.get_lines()[1].set_color('red')  # Color theoretical quantile line red
    plt.suptitle(f"Q-Q plot: {name} utan {to_remove} tidpunkter\nmed de lägsta {changed_by.lower()} värder", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'img/q_q_plot/{re.sub("(?i)"+re.escape("flygplats"), "", name)}_{changed_by.lower()}_min_{to_remove}_outliers.png')
    #plt.show()
    plt.close()

    """
    CORRELATION BETWEEN TEMP AND HUMIDITY FOR ALL DATA
    """
    # Pivote three_days
    pivote_df = df_three.pivot_table(index=['time', 'station_name'], columns='parameter', values='value').reset_index()


    sns.pairplot(pivote_df, hue='station_name')
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Parvisa relationer mellan temperatur och luftfuktighet", fontsize=10, y=0.95)
    plt.savefig("img/correlations/param_param.png")
    #plt.show()
    plt.close()

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
# Name of columns
column_name1 = "TEMPERATUR_Umeå Flygplats"
column_name2 = "LUFTFUKTIGHET_Umeå Flygplats"

# Pairplot 
fig = sns.pairplot(combined_data, height=1.8, diag_kind='kde')
# Adjust igont size and axis
for ax in fig.axes.flatten():
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
    
    # Set font size for tick labels
    ax.tick_params(axis='x', labelsize=5)  # X-axis tick labels
    ax.tick_params(axis='y', labelsize=5)  # Y-axis tick labels

plt.suptitle("Parvisa relationer för parametrar och stationer", y=0.99, fontsize=16)  # Title for the plot
plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9) # Ajust spase between subplots
plt.savefig('img/correlations/all_pairwise_relationships.png')
#plt.show()
plt.close()

"""
CORRELATION COEFFITIENTS
"""
# Calculate the correlation matrix
correlation_matrix = combined_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    correlation_matrix, 
    annot=True,  # Avoid cluttering with too many annotations
    cmap=LinearSegmentedColormap.from_list("CustomCmap", COLORS, N=256), 
    cbar=True
)
# Adjust font sizes for station names
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.title("Korrelationsmatris för alla parametrar och stationer", fontsize=14)
plt.tight_layout()
plt.savefig('img/correlations/all_correlations.png')
#plt.show()
plt.close()

"""
REGRESSION
"""
#Calculate regression for FUKT dependently on TEMP in UMEÅ
column_name1 = "TEMPERATUR_Umeå Flygplats"
column_name2 = "LUFTFUKTIGHET_Umeå Flygplats" 
# X (independent variable) and y (dependent variable)
X_data = combined_data[column_name1].values.reshape(-1, 1)  # Reshape for a single feature
y_data = combined_data[column_name2].values

# Initialize the model
model = LinearRegression()

# Fit data to the model
model.fit(X_data, y_data)

# Get the regression line parameters (slope and intercept)
slope = model.coef_[0] # =b
intercept = model.intercept_ # =a

# Print the regression equation
equation = f"Regression Equation: y = {intercept:.2f} + {slope:.2f} * X"

# Plot the data and regression line
plt.scatter(X_data, y_data, color='blue', label='Data Points')  # Plot the data points
plt.plot(X_data, model.predict(X_data), color='red', label='Regression Line')  # Plot the regression line

# Customize the plot
plt.xlabel(column_name1)
plt.ylabel(column_name2)
plt.title(f"Linear regression model.\nPrognos av relativt luftfuktighet på grund of temperatur i Umeå")

# label for the plot
line_kws={'color': 'red', 'label': f'Y = {intercept:.2f} + {slope:.2f}X'}
sns.regplot(x=column_name1, y=column_name2, data=combined_data, scatter_kws={'s': 10}, line_kws=line_kws)
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
LINIAR REGRESSION MODEL
"""
# Fraction of data to train
fraktion = 0.5
train = combined_data.sample(frac=fraktion, random_state=1)
test = combined_data.drop(train.index)

# Extract X (independent variable) and y (dependent variable)
X_train = train[column_name1].values.reshape(-1, 1)
y_train = train[column_name2].values
X_test = test[column_name1].values.reshape(-1, 1)
y_test = test[column_name2].values

# Trainings model
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

# MSE of test data
mse = np.mean((pred - y_test) ** 2)
b = model.coef_[0]
a = model.intercept_

# Use statsmodel for confidens interval
X_train_with_const = sm.add_constant(X_train)  # Lägg till konstant för intercept
ols_model = sm.OLS(y_train, X_train_with_const).fit()
conf_int_params = ols_model.conf_int(alpha=0.05)  # 95% konfidensintervall

# calculate confidens interval
intercept_ci = conf_int_params[0]  # Första raden: Intercept
slope_ci = conf_int_params[1]  # Andra raden: Lutning


plt.figure(figsize=(10, 6))
# Train data
plt.scatter(X_train, y_train, color="orange", label='Träningsdata', alpha=0.6)
# Test data
plt.scatter(X_test, y_test, color="blue", label='Testdata', alpha=0.6)

# Title
sns.regplot(x=column_name1, y=column_name2, data=combined_data, scatter=False, 
            line_kws={'color': 'red', 'label': f'Y = {a:.2f} + {b:.2f} * X'}, ci=95)
x_plot = np.linspace(-20, -1, 100).reshape(-1,1)
draw_plot = model.predict(x_plot)
# Regression line for predictions (testdata)
plt.plot(x_plot, draw_plot, color='g', label='Test Data Prediction', linewidth=2)

# show Regression equation and confidence interval
plt.text(0.5, 0.89, 
        f'Lineärmodel: y ={a:.2f} +  {b:.2f}* X\n'
        f'95% konfidens interval för a: [{intercept_ci[0]:.2f}, {intercept_ci[1]:.2f}]\n'
        f'95% konfidens interval för b: [{slope_ci[0]:.2f}, {slope_ci[1]:.2f}]', 
        ha='center', va='center', transform=plt.gca().transAxes, fontsize=10, color='red')

plt.title(f"Prognos av luftfuktighet baserat på temperatur i Umeå\nMean squared error: {mse:.2f}\nFraktion: {fraktion}")
plt.xlabel("Temperatur, °C")
plt.ylabel("Relativt Luftfuktighet, %")
plt.legend(loc='best')
plt.savefig(f'img/regression/Conf_int_regr_prediction_Umea_temp_luft.png')
#plt.show()
plt.close()

# Calculate residuals of test data
residuals = pred - y_test

# Calculate standarddeviation of residuals
std_residual = np.std(residuals)
stat, p = sci.shapiro(residuals)
# Visualise residuals
plt.scatter(X_test, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residualer av relativt luftfuktighet i Umeå")
plt.xlabel("temperatur i Umeå, °C")
plt.ylabel("Residualer av relativt luftfuktighet, %")
text = (f"Standard avvikelse för residualer: {std_residual:.2f} %\n"
        f"Shapiro-Wilk test for residualernas spridning: p = {p:.3f}, \nstatistics={stat:.3f}\n"
        f"Som visar att resudualernas spridning kan inte bearbetas som normal.")
# Place text at a specific location in data coordinates
plt.text(-20.0, 4.5, text, 
        color="green", fontsize=8, ha='left', va='bottom')
plt.savefig('img/regression/residuals_temp_fukt_UME.png')
#plt.show()
plt.close()

# Show histogram av residualen för test data
plt.hist(residuals, bins=24, edgecolor='white', alpha=1.0)
plt.title("Histogram av residualer av luftfuktighet i Umeå", fontsize=10)
plt.xlabel("Residuals")
plt.ylabel("Frekvens")
plt.savefig('img/regression/residuals_hist_temp_fukt_UME.png')
#plt.show()
plt.close() 

"""
MODIFATED DATA
LOG TRANSFORMATION temperatur och fuktighet i Umeå
"""
# Log transformation
# I cannot use direct log-transformaton due to negative values of the tempture
# I want to shift all values that the lowest is just above zero
X_combined = combined_data[column_name1].values
# shift all values above zero, 1e-5 ensure that no values are zero
shift_value = abs(X_combined.min()) + 1e-5
def log_stat_plot(x_tr, x_te, y_tr, y_te, name, title=""):
    X_train_log = np.log(x_tr + shift_value)
    X_test_log = np.log(x_te + shift_value)

    # Visualise log transformation
    plt.figure(figsize=(12, 6))
    # Show original data on subplot 1
    plt.subplot(1, 2, 1)
    plt.scatter(x_tr, y_tr, label='Träningsdata')
    plt.scatter(x_te, y_te, label='Test data')
    plt.legend()
    plt.xlabel("Temperatur, °C", fontsize=8)
    plt.ylabel("Relativt luftfuktighet; %", fontsize=8)
    plt.title("Skatterplot med originala x-värde", fontsize=10)

    # Show log transformation on subplot 2
    plt.subplot(1,2,2)
    plt.scatter(X_train_log, y_tr, label='Träningsdata')
    plt.scatter(X_test_log, y_te, label='Test data')
    plt.legend()
    plt.xlabel("Temperatur, log(value - min - 0.00001 )", fontsize=8)
    plt.ylabel("Relativt luftfuktighet; %", fontsize=8)
    plt.title("Skatterplot med log-transformerade x-värde", fontsize=10)
    plt.suptitle(title, fontsize=12)
    plt.savefig(f'img/regression/{name}.png')
    #plt.show()
    plt.close()
    return X_train_log, X_test_log
X_train_log, X_test_log = log_stat_plot(x_tr=X_train,
                                        x_te=X_test,
                                        y_tr=y_train,
                                        y_te=y_test, 
                                        name='residuals_log_data',
                                        title="Öforändrade data"
                                        )
"""
MODIFY DATASET BY REMOVING OUTLIERS
"""
# Sort the values to identify outliers along axis 0 (values)
X_combined_sotred = np.sort(X_combined, axis=0)

# Create array of values that sould be removed
X_to_remove = X_combined_sotred[:2].tolist() + X_combined_sotred[-2:].tolist()
X_to_remove = set(X_to_remove)
# fined what rows should be removed
rows_to_remove = combined_data[column_name1].apply(lambda x: any(np.isclose(x, value) for value in X_to_remove))
# Filter this rows away
filtered_data = combined_data.loc[~rows_to_remove]
train_filt = filtered_data.sample(frac=fraktion, random_state=1)
test_filt = filtered_data.drop(train_filt.index)

# Create new train and test dataset:
X_train_f = train_filt[column_name1].values.reshape(-1, 1)
y_train_f = train_filt[column_name2].values
X_test_f = test_filt[column_name1].values.reshape(-1, 1)
y_test_f = test_filt[column_name2].values
X_train_f_log, X_test_f_log = log_stat_plot(X_train_f,
                                            X_test_f,
                                            y_train_f,
                                            y_test_f,
                                            'residuals_log_data_filtered',
                                            title="Data utan två tidspunkter med de högsta och\ntva" + 
                                                "tidpunkter med de lägsta temperaturvärde"
                                            )

# Use filtered data to create the regression with log tempture
log_model = LinearRegression()
log_model.fit(X_train_f_log, y_train_f)

log_a = log_model.coef_[0]
log_b = log_model.intercept_

# Make prediction with test data
pred_log = log_model.predict(X_test_f_log)

x_log = np.linspace(-1.0, 3.01, 100)
draw_exp_model = log_model.predict(x_log.reshape(-1, 1))

# Calculate MSE for logarithmik transformation
mse_log = np.mean((pred_log - y_test_f)**2)

# Transform predictions back to the original scale
pred_original = np.exp(pred_log) - shift_value

# Calculate MSE in the original domain
mse_original = np.mean((pred_original - X_test_f)**2)

# Show prediktions
plt.scatter(X_train_f_log, y_train_f, label='Träningsdata')
plt.scatter(X_test_f_log, y_test_f, label='Test data')
plt.plot(x_log, draw_exp_model, label='Linjär regression, log transformerad i x', color='c', linewidth=3)
plt.text(-0.7, 90.0, f"y = {log_a:.2f} + {log_b:.2f}*log(x)", fontsize=8, color="r")
plt.legend()
plt.title("Prognos av relativt luftfuktighet med logaritmisk model (x är log transformerad)", fontsize=10)
plt.xlabel("Temperatur [log( °C)]", fontsize=8)
plt.ylabel("Relativt luftfuktighet, %", fontsize=8)
plt.savefig('img/regression/prediction_log_data.png')
#plt.show()
plt.close()

# Calculate residuals of test data
residual = y_test_f - pred_log
# this array contain nan becouse of previouse manipulations with data
# I want to remove nan elements:
data_without_nan = [x for x in residual if not math.isnan(x)]
std_residual = np.std(data_without_nan)

# Sheow residuals of the test data
plt.scatter(X_test_f, residual)
plt.axhline(0, color='r', linestyle='--')
plt.title("Logmodel predictions residulaer av luftfuktighet vid Umeå flygplats")
plt.xlabel("trelativt luftfuktighet,  %")
plt.ylabel("Residualer, %")
plt.savefig('img/regression/residuals_filtrerad_LOGtemp_fukt_UME.png')
#plt.show()
plt.close()

# Show histogram of the residuals of the test data
plt.hist(residual, bins=10)
plt.title("Residualernas histogram för logoritmisk model av luftfuktighet vid Umeå", fontsize=10)
plt.xlabel("Residualer av luftfuktighet, %")
plt.ylabel("Frekvens")
plt.savefig('img/regression/residuals_filtrerad_hist_LOGtemp_fukt_UME.png')
#plt.show()
plt.close() 

"""_summary_
"""
# get liniar model with the modyfiyed data:
# Trainings model
lin_model_f = LinearRegression().fit(X_train_f, y_train_f)
lin_pred_f = model.predict(X_test_f)

# MSE of test data
mse_lin_f = np.mean((lin_pred_f - y_test_f) ** 2)
# Transform back model
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(np.exp(x_log) - shift_value, draw_exp_model, label='Linjär regression, exponentiell i x', color='c', linewidth=3)
plt.legend()
plt.title("Prognos av relativt luftfuktighet med logaritmisk model", fontsize=10)
plt.xlabel("temperatur, °C", fontsize=8)
plt.ylabel("relativt luftfuktighet, %", fontsize=8)

text = (f"MSE(mean squared error) för liniarregressoin : {mse:.2f}\n"
    f"MSE för liniarregressoin utan avvikande värde:'.rjust(60): {mse_lin_f:.2f}\n"
    f"MSE för logoritmisk model utan avvikande värde: {mse_log:.2f}")
plt.text(-15.0, 80.8, text, color="red", fontsize=8)
plt.savefig('img/regression/log_transform_back.png')
#plt.show()
plt.close()

"""
Y_LOG
"""
# Regression with log relativt humidity
# Relative humidity is always pozitiv
y_train_log = np.log(y_train_f)
y_test_log = np.log(y_test_f)

# create liniar regression pf the log data
log_y_model = LinearRegression()
log_y_model.fit(X_train_f, y_train_log)

# Make prediction of the test data
x_for_log_y = np.linspace(-1.0, 3.01, 100)
pred_log_y = log_y_model.predict(X_test_f)

b_Y_log = log_y_model.coef_[0]
a_Y_log = log_y_model.intercept_

x_for_log_y = np.linspace(-25,-3, 100)
y_log = log_y_model.predict(x_for_log_y.reshape(-1, 1))

# Calculate MSE
mse_log_y = np.mean((np.exp(pred_log_y) - y_test_f)**2)
plot_text= (f"MSE linjär regression: {mse:.2f}\n"
            f"MSE för liniarregressoin utan avvikande värde: {mse_lin_f:.2f}\n"
            f"MSE logtransformerad x: {mse_log:.2f}\n"
            f"MSE logtransformerad y: {mse_log_y:.5f}")


plt.scatter(X_train_f, y_train_log, label='Träningsdata')
plt.scatter(X_test_f, y_test_log, label='Test data')
plt.plot(x_for_log_y, y_log, label='Linjär regression log domän', color='g', linewidth=3)
plt.legend()
plt.text(-24, y_test_log[5], f"log(Y) = {a_Y_log:.2f} + {b_Y_log:.2f} * X", fontsize=8, color="red")
plt.title("Prognos av relativt luftfuktighet (log transformerad y, %)")
plt.xlabel("temperatur, °C")
plt.ylabel("relativt luftfuktighet [log %]")
plt.savefig('img/regression/log_transform_FUKT_Umeå.png')
plt.show()
plt.close()


# Calculate residualer
residual_log_y = y_test_f - np.exp(pred_log_y)

p_log_y, stat_log_y = sci.shapiro(residual_log_y)

print(p_log_y)
print(stat_log_y)

#Coordinats for the plot axis
x = np.linspace(-22, -0.5, 100)
y_log = log_y_model.predict(x.reshape(-1, 1))
# VShow exponential modell on y
plt.scatter(X_train_f, y_train_f, label='Träningsdata')
plt.scatter(X_test_f, y_test_f, label='Test data')
plt.plot(x, np.exp(y_log), label='Linjär regression exponentiell i y', color='g', linewidth=3)
plt.legend()
plt.title("Prognos av relativt luftfuktighet (exponentiell modell i y)")
plt.ylabel("Relativt lutfuktighet %")
plt.xlabel("Temperatur, °C")
plt.text(-22.5, 90.5, plot_text, fontsize=8, color="r")
plt.savefig('img/regression/Y_LOG_transform_model_Umeå.png')
plt.show()
plt.close()

#ALLA MODELLER PÅ EN PLOT
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(x, np.exp(y_log), label='Linjär regression exponentiell i y', color='orange', linewidth=3)
plt.plot(np.exp(x_log) - shift_value, draw_exp_model, label='Linjär regression, exponentiell i x', color='red', linewidth=3)
plt.plot(x_plot, pred, color='green', label='Test Data Prediction', linewidth=2)
plt.title("Prognos av relativt luftfuktighet med alla modeller")
plt.ylabel("Relativt lutfuktighet %")
plt.xlabel("Temperatur, °C")
plt.legend(loc='best', frameon=True, fontsize=10, title="Modeller och data", title_fontsize=10)
plt.tight_layout()
plt.savefig('img/regression/alla_modeller_Umeå.png')
plt.show()
plt.close()