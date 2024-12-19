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
import seaborn as sns
## own functions
import utils
# variables
STATIONS = {'Halmstad flygplats': 62410, 'Uppsala Flygplats': 97530, 'Umeå Flygplats': 140480}
# Directory to save the data files and statistics
OUTPUT_DIR = {"data":"smhi_data_temp_fukt", "img":"img", "statistics":"statistics"}
#os.makedirs(OUTPUT_DIR["data"], exist_ok=True)

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
all_data = {}
three_days = {}

for param_id, parameter in PARAMS.items():
    station_data = {}
    three_d_station = {}
    for name, station_id in STATIONS.items():
        file_path = OUTPUT_DIR["data"] + '/' + f'{station_id}_{param_id}.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
            station_data[name] = data
            # Extract the "value" list and sort it by timestamp
            sorted_data = sorted(
                data.get("value", []),
                key=lambda x: datetime.fromtimestamp(x["date"] / 1000, tz=pytz.timezone("Europe/Stockholm"))
            )
            # Get the last N points
            last_points = sorted_data[-mesured_points:]
        """the arrays' item are dict with keys: date, value and quality. 
        I want remove quality but replace value to nympy.nan if quality is not G or Y
        """
        stat_set = {}
        for item in last_points:
            new_value = float(item['value']) if item['quality'] in ['G', 'Y'] else np.nan
            stat_set[item['date']] = new_value  # Add date-value pair to value_set
    
        three_d_station[name] = stat_set
    all_data[param_id] = station_data
    three_days[param_id] = three_d_station
# chage dataset for each parameter into pandas DateFrame
for key in three_days.keys():
    
    df = pd.DataFrame.from_dict(three_days[key], orient='columns')
    # convert time index from timestamp to date-time format
    df.index = pd.to_datetime(df.index, unit='ms')
    # Make it readable string
    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    #summary of missing data: 
    missing_summary = df.isna().sum()
    missing_summary_df = missing_summary.to_frame().T  # Convert Series to DataFrame (single-row)
    md_tabel = utils.change_to_markdown(missing_summary, None)
    utils.save_to_mdfile(md_tabel, f'{PARAMS[key][0]}_mis_summ.md', OUTPUT_DIR['statistics'])
    # Generate statistics for each location
    stats = df.describe().round(2)
    md_tabel = utils.change_to_markdown(stats, None)
    
    utils.save_to_mdfile(md_tabel, f'{PARAMS[key][0]}_describe_stat.md', OUTPUT_DIR['statistics'])
   
"""
Distributions plottar för enskilda parametrar
"""
# create plot for each parameter
for param_id in three_days.keys():
    df = pd.DataFrame.from_dict(three_days[param_id], orient='columns')
    norm_distr = utils.stat_norm_distribution(df)
    
    # Define Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a single figure with multiple subplots (one for each station)
    num_columns = len(df.columns)  # Number of stations
    fig, axes = plt.subplots(1, num_columns, figsize=(3 * num_columns, 4), squeeze=False)  # Subplots as a 2D array
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    x_lable = PARAMS[param_id][0].lower() + " " + PARAMS[param_id][1]
    
    utils.define_axix(axes, x_lable,  "frekvens", df, array2=norm_distr)
    # Adjust spacing
    plt.tight_layout()
    
    # Save combined plot
    path = f'img/frekvenser/{PARAMS[param_id][0]}_combined.png'
    plt.savefig(path)

    # # # Show the combined plot
    # # plt.show()
    plt.close()
    
# Create Q_Q plot for each parametr
for param_id in three_days.keys():    
    
    """
    Plots Q-Q plots for each column in the dataframe to check normality visually.
    All Q-Q plots are displayed in a single figure with multiple subplots.
    """
    df = pd.DataFrame.from_dict(three_days[param_id], orient='columns')
    norm_distr = utils.stat_norm_distribution(df)
    # Determine the number of columns (stations)
    num_columns = len(df.columns)
    # Define Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a single figure with multiple subplots (one for each column)
    fig, axes = plt.subplots(1, num_columns, figsize=(4 * num_columns, 4), squeeze=False)  # Subplots as a 2D array
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    # Loop through each column in the DataFrame
    for ax, key in zip(axes, df.columns):
        values = df[key]  # Extract the column values
        
        # Generate Q-Q plot for each station/column
        sci.probplot(values, dist="norm", plot=ax)
        
        # Title and labels for the plot
        ax.set_title(f"Q-Q Plot för {key}", fontsize=10)  # Title for the plot
        ax.set_xlabel("Teoretiska kvantiler", fontsize=10)
        ax.set_ylabel(f"Ordnade {PARAMS[param_id][0]} kvantiler, {PARAMS[param_id][1]}", fontsize=10)
        ax.grid()

    # Adjust spacing between subplots
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5) 

    # # Adjust the spacing between the subplots (increase space between them)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the width and height spacing between plots
    # Save the combined plot
    path = f'img/q_q_plot/{PARAMS[param_id][0]}_combined_qq_plots.png'
    plt.savefig(path)
    print(f"Combined Q-Q Plot saved to {path}")
    
    # Show the combined plot
    #plt.show()
    plt.close()
    
# # distribution for all month set data
# for param_id in all_data.keys():    
    
#     """
#     Plots Q-Q plots for each column in the dataframe to check normality visually.
#     All Q-Q plots are displayed in a single figure with multiple subplots.
#     """
#     df = pd.DataFrame.from_dict(all_data[param_id], orient='columns')
#     norm_distr = utils.stat_norm_distribution(df)
#     # Determine the number of columns (stations)
#     num_columns = len(df.columns)
#     # Define Seaborn style
#     sns.set_theme(style="whitegrid")
    
#     # Create a single figure with multiple subplots (one for each column)
#     fig, axes = plt.subplots(1, num_columns, figsize=(4 * num_columns, 4), squeeze=False)  # Subplots as a 2D array
    
#     # Flatten axes for easy iteration
#     axes = axes.flatten()
#     # Loop through each column in the DataFrame
#     for ax, key in zip(axes, df.columns):
#         values = df[key]  # Extract the column values
        
#         # Generate Q-Q plot for each station/column
#         sci.probplot(values, dist="norm", plot=ax)
        
#         # Title and labels for the plot
#         ax.set_title(f"Q-Q Plot för 1000 timmar data of {key}", fontsize=10)  # Title for the plot
#         ax.set_xlabel("Teoretiska kvantiler", fontsize=10)
#         ax.set_ylabel(f"Ordnade {PARAMS[param_id][0]} kvantiler, {PARAMS[param_id][1]}", fontsize=10)
#         ax.grid()

#     # Adjust spacing between subplots
#     plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5) 

#     # # Adjust the spacing between the subplots (increase space between them)
#     # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the width and height spacing between plots
#     # Save the combined plot
#     path = f'img/q_q_plot/{PARAMS[param_id][0]}_combined_1000h_qq_plots.png'
#     plt.savefig(path)
#     print(f"Combined Q-Q Plot saved to {path}")
    
#     # Show the combined plot
#     #plt.show()
#     plt.close()
