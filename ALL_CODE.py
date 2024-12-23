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

            # Append each row with the timestamp, station, parameter_id, and value
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

def comment_avay():
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

    # Iterate through all stations and parameters
    for i, station in enumerate(stations):
        for j, parameter in enumerate(parameters):
            # filter data for each station and parameter
            data = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]

            # Subplot indexering: 3 rows for 3 stations and 2 columns for 2 parameters
            plt.subplot(3, 2, i * len(parameters) + j + 1) 
            
            # create histogramm
            sns.histplot(data['value'], kde=True, bins=24, color="green", edgecolor="black")
            
            # add title and axes
            plt.title(f"{station} - {parameter}")
            plt.xlabel("Värde")
            plt.ylabel("Frekvens")

    plt.tight_layout()
    plt.savefig("img/frekvenser/alla.png")

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
            # Filter data for the current station and parameter
            data_filtered = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]
            stat, p_value = sci.shapiro(data_filtered['value'])
            results.append({
                'Station': station,
                'Parameter': parameter,
                'Shapiro-Wilk Statistic': round(stat, 5),
                'P-value': round(p_value, 5),
                'Normal Distribution (p > 0.05)': 'Yes' if p_value > 0.05 else 'No'
            })
            
            # Select the current axis
            ax = axes[i, j]
            
            # Plot the boxplot
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

            # Add title and labels
            ax.set_title(f"{station} - {parameter}", fontsize=8)
            #ax.set_xlabel("Station Name", fontsize=10)
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

    # Adjust layout
    plt.tight_layout()

    # Show the figure
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


# Loopa igenom alla stationer och parametrar
for i, station in enumerate(stations):
    for j, parameter in enumerate(parameters):
        # Filtrera data för station och parameter
        data = df_three[(df_three['station_name'] == station) & (df_three['parameter'] == parameter)]
        numeric_data = data['value'].dropna()
        # Skapa Q-Q diagram för att jämföra med normalfördelning
        ax = axes[j, i]
        
        sci.probplot(numeric_data, dist="norm", plot=ax)
        ax.set_ylabel(f"{'temperatur, °C' if parameter == 'TEMPERATUR' else 'humidity, %'}", fontsize=8)
        # Lägg till titel
        ax.set_title(f"Q-Q plot: {station} - {parameter}", fontsize=8)
        ax.get_lines()[1].set_color('red')  # Gör linjen för teoretiska kvantiler röd
        
# Justera layouten
plt.tight_layout()
plt.savefig('img/q_q_plot/all.png')
plt.close()
exit()
"""
CORRELATION MATRIX FOR 6 PARAMETERS
"""
# Combine data for all parameters, with station names included
combined_data = pd.DataFrame()
column_name1 = "TEMPERATUR_Umeå Flygplats"  # Replace with actual column name for temperature
column_name2 = "LUFTFUKTIGHET_Umeå Flygplats"  # Replace with actual column name for humidity


for param_key, dataset in three_days.items():
    param_name = PARAMS[param_key][0]  # Parameter name (e.g., Temperature, Humidity)
    df = pd.DataFrame(dataset)
    df.columns = [f"{param_name}_{station}" for station in df.columns]  # Add station to column names
    combined_data = pd.concat([combined_data, df], axis=1)
# Calculate the correlation matrix
correlation_matrix = combined_data.corr()
def functio_sceep():
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
    plt.close()
    #
    # Use pairplot to visualize pairwise relationships
    f = sns.pairplot(combined_data, height=1.8, diag_kind='kde')
    # Adjust font size for axis labels, titles, and ticks
    for ax in f.axes.flatten():
        # Get current x and y axis labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        # Apply all transformations to the xlabel and ylabel in one go:
        xlabel = re.sub(r'(?i)flygplats', '', xlabel).strip()  # Remove "flygplats" (case-insensitive)
        xlabel = xlabel.replace("TEMPERATUR_", "TEMP_").strip()  # Replace "TEMPERATUR_" with "TEMP_"
        xlabel = xlabel.replace("LUFTFUKTIGHET_", "FUKT_").strip()  # Replace "LUFTFUKTIGHET_" with "FUKT_"
        
        ylabel = re.sub(r'(?i)flygplats', '', ylabel).strip()  # Remove "flygplats" (case-insensitive)
        ylabel = ylabel.replace("TEMPERATUR_", "TEMP_").strip()  # Replace "TEMPERATUR_" with "TEMP_"
        ylabel = ylabel.replace("LUFTFUKTIGHET_", "FUKT_").strip()  # Replace "LUFTFUKTIGHET_" with "FUKT_"
        
        # Set the modified labels with font size
        ax.set_xlabel(xlabel, fontsize=6)
        ax.set_ylabel(ylabel, fontsize=6)
        
        ax.set_ylabel(ylabel.replace("LUFTFUKTIGHET_", "FUKT_").strip(), fontsize=6)
        
        # Set font size for tick labels
        ax.tick_params(axis='x', labelsize=5)  # X-axis tick labels
        ax.tick_params(axis='y', labelsize=5)  # Y-axis tick labels

    plt.suptitle("Pairwise Relationships for Parameters and Stations", y=0.99, fontsize=16)  # Title for the plot
    plt.subplots_adjust(hspace=0.2, wspace=0.2, top=0.9) # Ajust spase between subplots
    plt.savefig('img/regression/all_pairwise_relationships.png')
    # plt.show()
    plt.close()


    """
    Calculate regression for FUKT dependently on TEMP in UMEÅ
    """
    # Specify the column names
    
    # Extract X (independent variable) and y (dependent variable) from the dataframe
    X = combined_data[column_name1].values.reshape(-1, 1)  # Reshape for a single feature
    y = combined_data[column_name2].values  # Dependent variable (y)

    # Initialize the LinearRegression model
    model = LinearRegression()

    # Fit the model on the data
    model.fit(X, y)

    # Get the regression line parameters (slope and intercept)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Print the regression equation
    print(f"Regression Equation: y = {slope:.2f} * X + {intercept:.2f}")

    # Plot the data and regression line
    plt.scatter(X, y, color='blue', label='Data Points')  # Plot the data points
    plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Plot the regression line

    # Customize the plot
    plt.xlabel(column_name1)
    plt.ylabel(column_name2)
    plt.title(f"Linear regression model.\nPrediktion av relativt luftfuktighet på grund of temperatur i Umeå")

    # Create the regression plot with a custom label
    sns.regplot(x=column_name1, y=column_name2, data=combined_data, scatter_kws={'s': 10}, line_kws={'color': 'red', 'label': f'Y = {slope:.2f}X + {intercept:.2f}'})
    # Manually handle the legend and remove unwanted "Regression Line" entry
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

def sceep_2():
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    # Räkna ut MSE
    mse = np.mean((pred - y_test)**2)
    linear_slope = model.coef_[0]
    linear_intercept = model.intercept_


    # Add linear regression parameters to the plot
    plt.text(0.5, 0.95, f'Linear Model: y = {linear_slope:.2f}x + {linear_intercept:.2f}',
            ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')
    # Visulisera prediktioner
    plt.scatter(X_train, y_train, label='Träningsdata')
    plt.scatter(X_test, y_test, label='Test data')
    plt.plot(X_test, pred, label='Linjär regression', color='g', linewidth=3)
    plt.legend()
    plt.title(f"Prediktioner av luftfuktighet på temperatur i Umeå\nMean squared error: {mse}" + 
            f"\nFraktion: {fraktion}")
    plt.xlabel("Temperatur")
    plt.ylabel("Relativt Luftfuktighet")
    path = f'img/regression/regr_prediction_Umea_temp_luft_{fraktion}.png'
    plt.savefig(path)
    # plt.show()
    plt.close() 


    # Beräkna residualen för test data
    residual = y_test - pred

    # Beräkna standardavvikelsen för residualen
    std_residual = np.std(residual)
    #print(f"Standardavvikelsen för residualen: {std_residual:.2f}")

    # Visualisera residualen för test data
    plt.scatter(X_test, residual)
    plt.axhline(0, color='r', linestyle='--')
    plt.title("Residualer av temperatur i Umeå")
    plt.xlabel("temperatur i Halmstad")
    plt.ylabel("Residual")
    path = f'img/regression/residuals_temp_fukt_UME.png'
    plt.savefig(path)
    plt.close()

    # Visa histogram av residualen för test data
    plt.hist(residual, bins=10)
    plt.title("Histogram av residualer av luftfuktighet i Umeå")
    plt.xlabel("Residual")
    plt.ylabel("Frekvens")
    path = f'img/regression/residuals_hist_temp_fukt_UME.png'
    # plt.savefig(path)
    # plt.show()
    plt.close() 

"""

MODIFIERA DATA

TRANSFORMATION temperatur och fuktighet i Umeå
"""

# Logoritmisk transformation
# Jag kan inte använda direkt logaritmisk transformation för temperatur, pga negativa value 

X_combined = combined_data[column_name1].values

shift_value = abs(X_combined.min()) + 1e-5  # Ensure no zero values

X_train_log = np.log(X_train + shift_value)
X_test_log = np.log(X_test + shift_value)

# Visualisera logaritmisk transformation
plt.figure(figsize=(12, 6))

# Visa original data i subplot 1
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.legend()
plt.title("Original data")
plt.xlabel("Temperatur")
plt.ylabel("Relativt luftfuktighet")

# Visa logaritmisk transformation i subplot 2
plt.subplot(1, 2, 2)
plt.scatter(X_train_log, y_train, label='Träningsdata')
plt.scatter(X_test_log, y_test, label='Test data')
plt.legend()
plt.title("Logaritmisk transformation")
plt.xlabel("Temperatur, log(value - min - 0.00001 )")
plt.ylabel("Relativt luftfuktighet")
plt.savefig(f'img/regression/log_data.png')
#plt.show()
plt.close()

# Bygg linjär regression med logaritmerad temperatur
log_model = LinearRegression()
log_model.fit(X_train_log, y_train)

# Prediktera med test data
pred_log = log_model.predict(X_test_log)

x_log = np.linspace(-0.2, 3.0, 100)
draw_exp_model = log_model.predict(x_log.reshape(-1, 1))

# Räkna ut MSE
mse_log = np.mean((pred_log - X_test_log)**2)
print("Mean squared error log transformerad:", mse_log)
# Transform predictions back to the original scale
pred_original = np.exp(pred_log) - shift_value

# Calculate MSE in the original domain
mse_original = np.mean((pred_original - X_test)**2)
print("Mean squared error original domain:", mse_original)
#print("Mean squared error linjär regression:", mse)

# Visalisera prediktioner
plt.scatter(X_train_log, y_train, label='Träningsdata')
plt.scatter(X_test_log, y_test, label='Test data')
plt.plot(x_log, draw_exp_model, label='Linjär regression, log transformerad i x', color='c', linewidth=3)
plt.legend()
plt.title("Prediktioner av relativt luftfuktighet (log transformerad)")
plt.xlabel("Temperatur [log]")
plt.ylabel("Relativt luftfuktighet")
plt.savefig(f'img/regression/residuals_log_data.png')
#plt.show()
plt.close()

# Beräkna residualen för test data
residual = y_test - pred_log

# Beräkna standardavvikelsen för residualen
std_residual = np.std(residual)

# Visualisera residualen för test data
plt.scatter(X_test, residual)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residualer av temperatur i Umeå")
plt.xlabel("temperatur i Halmstad")
plt.ylabel("Residual")
path = f'img/regression/residuals_LOGtemp_fukt_UME.png'
plt.savefig(path)
# plt.show()
plt.close()

# Visa histogram av residualen för test data
plt.hist(residual, bins=10)
plt.title("Histogram av residualer av log_temperatur prediction av luftfuktighet i Umeå")
plt.xlabel("Residual")
plt.ylabel("Frekvens")
path = f'img/regression/residuals_hist_LOGtemp_fukt_UME.png'
plt.savefig(path)
#plt.show()
plt.close() 

# Transformera tillbaka modellen
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(np.exp(x_log) - shift_value, draw_exp_model, label='Linjär regression, exponentiell i x', color='c', linewidth=3)
plt.legend()
plt.title("Prediktioner av relativt luftfuktighet exponentiell modell")
plt.xlabel("Temperatur")
plt.ylabel("Relativt luftfuktighet")
plt.savefig('img/regression/transform_back.png')
#plt.show()
plt.close()

# Regression med logaritmerad relativt luft fuktighet
# Relativt luftfuktighet är alltid pozitivt
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Bygg linjär regression av logaritmerad data
log_y_model = LinearRegression()
log_y_model.fit(X_train, y_train_log)

# Prediktera med test data
pred_log_y = log_y_model.predict(X_train)
# pred_log_y = np.exp(pred_log_y)

# Räkna ut MSE
mse_log_y = np.mean((pred_log_y - y_test_log)**2)
print("Mean squared error log transformerad y:", mse_log_y)
print("Mean squared error log transformerad x:", mse_log)
#print("Mean squared error linjär regression:", mse)

# Visalisera prediktioner
x = np.linspace(-21, -0.5, 100)
y_log = log_y_model.predict(x.reshape(-1, 1))

plt.scatter(X_train, y_train_log, label='Träningsdata')
plt.scatter(X_test, y_test_log, label='Test data')
plt.plot(x, y_log, label='Linjär regression log domän', color='g', linewidth=3)
plt.legend()
plt.title("Prediktioner av relativt luftfuktighet (log transformerad y)")
plt.xlabel("Temperatyr")
plt.ylabel("Relativt luftfuktighet [log]")
plt.savefig('img/regression/log_transform_FUKT_Umeå.png')
#plt.show()
plt.close()

# Beräkna residualer
residual_log_y = y_test - np.exp(pred_log_y)

# Skriv ut MSE
mse_exp_y = np.mean(residual_log_y**2)
print(f"MSE av kronbladsbredd (exponentiell modell i y): {mse_exp_y}")
print(f"Mse av kronbladsbredd (log transformerad x): {mse_log}")
#print(f"Mse av kronbladsbredd (linjär regression): {mse}")

# Visualisera exponential modell i y
plt.scatter(X_train, y_train, label='Träningsdata')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(x, np.exp(y_log), label='Linjär regression exponentiell i y', color='g', linewidth=3)
plt.legend()
plt.title("Prediktioner av kronbladsbredd (exponentiell modell i y)")
plt.xlabel("Kronbladslängd")
plt.ylabel("Kronbladsbredd")
plt.savefig('img/regression/Y_LOG_transform_model_Umeå.png')
plt.show()