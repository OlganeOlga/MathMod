"""
this file preperes all for uppgift 1-2
"""
import scipy.stats as sci
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import get_dynam_data.prepere_data as p_d
# Skapa en egen färgskala
CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "CustomCmap", ["white", "yellow", "green"], N=256
)

""" input dictionary in form:
data = {
    'name': {
    timestamp: 'temp', ...,  timestamt: '2temp'
    },
    ...
 }
"""

def missing_data(df):
    #summary of missing data: 
    missing_summary = df.isna().sum()
    p_d.append_to_markdown(missing_summary)

#p_d.append_to_markdown(missing_summary_fukt)

def graph_missing_data():
    # get timestamp where data missin:
    for column in df.columns:
        missing_timestamps = df[df[column].isna()].index.tolist()
        print(f"{column} is missing data at timestamps: {missing_timestamps}")
    #wisualise missing data: 
    plt.figure(figsize=(10, 6))
    sns.heatmap(missing_data, cmap="coolwarm", cbar=False, yticklabels=True)
    plt.title("Missing Data Heatmap")
    plt.show()

    #procentage of missing data:
    missing_percentage = df.isna().mean() * 100
    print(missing_percentage)

def stat_norm_distribution(df: pd.DataFrame):
    """
    Plots the distribution (frequency) of temperatures across multiple stations.
    The x-axis represents the temperature values, and the y-axis represents the frequency of each temperature.
    """
    data= df.to_dict(orient="list")
    result = {}
    for key, value in data.items():
        print(value)
        stat, p_value = sci.shapiro(value)
        result[key] = [stat, p_value]
    return result

def plot_qq_plots(df: pd.DataFrame, param: str, unit="°C"):
    """
    Plots Q-Q plots for each column in the dataframe to check normality visually.
    All Q-Q plots are displayed in a single figure with multiple subplots.
    """
    # Determine the number of columns (stations)
    num_columns = len(df.columns)
    
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
        ax.set_ylabel(f"Ordnade {param} kvantiler, {unit}", fontsize=10)
        ax.grid()

    # Adjust spacing between subplots
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5) 

    # # Adjust the spacing between the subplots (increase space between them)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust the width and height spacing between plots
    # Save the combined plot
    path = f'img/distribution/{param}_combined_qq_plots.png'
    plt.savefig(path)
    print(f"Combined Q-Q Plot saved to {path}")
    
    # Show the combined plot
    plt.show()
    

# # Call the function
# plot_qq_plots(df=df)

def plot_param_distribution(df: pd.DataFrame, param="TEMPERATUR"):
    """
    Plots the temperature distributions for all stations in a single figure.
    The x-axis represents the temperature values, and the y-axis represents the frequency.
    """
    norm_distr = stat_norm_distribution(df)
    # print(norm_distr)
    # Define Seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create a single figure with multiple subplots (one for each station)
    num_columns = len(df.columns)  # Number of stations
    fig, axes = plt.subplots(1, num_columns, figsize=(3 * num_columns, 4), squeeze=False)  # Subplots as a 2D array
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, (key, values) in zip(axes, df.items()):
        sns.histplot(values, kde=True, bins=15, color='blue', edgecolor='black', ax=ax)
        
        # Add titles and labels
        ax.set_title(f'Frekvens spridning i {key}', fontsize=10)
        ax.set_xlabel(param.lower(), fontsize=12)
        ax.set_ylabel("frekvens", fontsize=12)
        print(key)
        # Add Shapiro-Wilk test results as annotation
        stat, p_value = norm_distr[key]
        ax.text(0.05, 0.95, f"Shapiro-Wilk:\nStat: {stat:.4f}\nP-value: {p_value:.4g}",
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    
    # Adjust spacing
    plt.tight_layout()
    
    # Save combined plot
    path = f'img/frekvenser/{param}_combined.png'
    plt.savefig(path)
    print(f"Plot saved to {path}")
    # save plot figure
    #plt.savefig(f"img/frekvens/all_{param}.png")

    # Show the combined plot
    plt.show()

# # # save unchanged copy timestamp to datatime:
# # df_timest = df.copy(deep=True)
# def timestamp_to_date(df=df):
#     # change timestamp to date
#     df.index = pd.to_datetime(df.index, unit="ms")
#     # use only hour
#     df["hour"] = df.index.hour


# # # Generate descriptive statistics
# # stats = df.describe()

# def value_on_hour(output_dir="grapfic_decription"):
#     os.makedirs(output_dir, exist_ok=True)
#     # Create a plot for temperature data (without modifying the original data)
#     plt.figure(figsize=(10, 6))
#     for column in df.columns:
#         # Extract the hours only for the rows with non-null values
#         valid_data = df[column].dropna()
        
#         # Use the index of valid_data directly to get the corresponding hours
#         valid_hours = valid_data.index.hour
        
#         # Handle missing values only in the plot, without changing the original data
#         plt.plot(valid_hours, valid_data, marker='o', label=column)            

#         #  Add labels and title
#         plt.xlabel('Hour of the day')
#         plt.ylabel('Temperature (°C)')
#         plt.xticks(range(24))  # Set x-ticks to range from 0 to 23 (hours of the day)
#         plt.grid(True)
#         plt.legend()
        
#         # save plot figure
#         plt.savefig(f"{output_dir}/{column}_temperature_plot.png")

#         # Show plot
#         plt.show()   
    # # Calculate the frequency of each unique temperature
    # temp_counts = all_temperatures.value_counts().sort_index()  # Sorted by temperature
    
    # # Create the scatter plot (Temperature vs Frequency)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(temp_counts.index, temp_counts.values, color='blue', alpha=0.7)
    
    # # Adding title and labels
    # plt.title('Distribution of Temperatures Across Stations (Frequency)')
    # plt.xlabel('Temperature (°C)')
    # plt.ylabel('Frequency')
    
    # # Display the grid for better readability
    # plt.grid(True)

    # # Show the plot
    # plt.show()

# Call the function to plot temperature distribution
#plot_temperature_distribution(df)


# def plot_missing_data_points():
#     # Create a DataFrame indicating where data is missing (True for missing, False for present)
#     missing_data = df.isna()
#     output_dir = ".grapfic_decription"
    
#     # Convert the index to datetime if not already done
#     if not pd.api.types.is_datetime64_any_dtype(df.index):
#         df.index = pd.to_datetime(df.index, unit='ms')
    
#     # Extract dates and hours from the index
#     dates = df.index.date
#     hours = df.index.hour
    
#     # Create a scatter plot for each station
#     fig, ax = plt.subplots(figsize=(12, 8))
#     for station in df.columns:
#         station_missing = missing_data[station]
        
#         # Filter dates and hours where data is missing
#         missing_dates = dates[station_missing]
#         missing_hours = hours[station_missing]
        
#         # Plot the missing data points for this station
#         ax.scatter(
#             missing_hours, missing_dates, label=station, alpha=0.7
#         )

#     # Add labels, legend, and grid
#     ax.set_title("Missing Data Points by Station")
#     ax.set_xlabel("Hour of the Day")
#     ax.set_ylabel("Date")
#     ax.set_xticks(range(24))  # Set x-ticks for all hours
#     ax.legend(title="Stations")
#     ax.grid(True, which="both", linestyle="--", alpha=0.5)

#     # Show the plot
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/missing_temperature_plot.png")
#     plt.show()

# # Call the function
# #plot_missing_data_points()


# def plot_missing_data_separate():
#     # Create a DataFrame indicating where data is missing (True for missing, False for present)
#     missing_data = df.isna()
    
#     # Convert the index to datetime if not already done
#     if not pd.api.types.is_datetime64_any_dtype(df.index):
#         df.index = pd.to_datetime(df.index, unit='ms')
    
#     # Extract dates and hours from the index
#     dates = df.index.date
#     hours = df.index.hour

#     # Create a separate plot for each station
#     for station in df.columns:
#         # Filter missing data for this station
#         station_missing = missing_data[station]
#         missing_dates = df.index[station_missing]  # Use full datetime index here
#         missing_hours = hours[station_missing]

#         # Plot the missing data
#         plt.figure(figsize=(10, 6))
#         plt.scatter(missing_hours, missing_dates, color='red', alpha=0.7, label='Missing Data')
        
#         # Customize the plot
#         plt.title(f"Missing Data for {station}")
#         plt.xlabel("Hour of the Day")
#         plt.ylabel("Date")
#         plt.xticks(range(24))  # Set x-ticks for all hours

#         # Set the date format for the Y-axis to show year, month, and day
#         ax = plt.gca()
#         ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#         ax.yaxis.set_major_locator(mdates.DayLocator(interval=2))  # Adjust interval as needed

#         plt.grid(True, which="both", linestyle="--", alpha=1)
#         plt.legend()

#         # Show the plot
#         plt.tight_layout()
#         plt.show()

# # Call the function
# #plot_missing_data_separate()

# def plot_missing_data_with_custom_axis():
#     # Create a DataFrame indicating where data is missing (True for missing, False for present)
#     missing_data = df.isna()
    
#     # Ensure the index is in datetime format
#     if not pd.api.types.is_datetime64_any_dtype(df.index):
#         df.index = pd.to_datetime(df.index, unit="ms")

#     # Extract hours from the index
#     hours = df.index.hour

#     # Plot for each station separately
#     for station in df.columns:
#         station_missing = missing_data[station]

#         # Extract only missing data points
#         missing_dates = df.index[station_missing]
#         missing_hours = hours[station_missing]

#         # Convert missing_dates to a pandas DatetimeIndex
#         missing_dates = pd.to_datetime(missing_dates)
        
#         # initiate extended_dates
#         extended_dates = list(missing_dates)
        
#         timestamps_series = pd.Series(extended_dates)
#         # Extract only the date part of the timestamps
#         dates_only = timestamps_series.dt.date

#         # Get unique dates (without duplicates)
#         unique_dates = pd.to_datetime(dates_only).unique()

#         # Extract only the date part of the timestamps
#         dates_only = timestamps_series.dt.date

#         # Get unique dates (without duplicates)
#         unique_dates = pd.to_datetime(dates_only).unique()

#         if "Halmstad" in station:
#             plt.figure(figsize=(8, 6))
#         elif "Umeå" in station:
#             # Create the figure
#             plt.figure(figsize=(7, 2))
#         elif "Upsala" in station:
#             # Create the figure
#             plt.figure(figsize=(4, 2))
       
#         if "Upsala" in station:
#             plt.scatter(missing_hours, [unique_dates[0]] * len(missing_hours), color="red", label="Missing Data", alpha=0.7)
#         else:
#             plt.scatter(missing_hours, missing_dates.date, color="red", label="Missing Data", alpha=0.7)

#         ax = plt.gca()

#         ax.set_yticks([pd.Timestamp(date) for date in extended_dates])
#          # Set y-axis labels with formatted date
#         ax.set_yticklabels([pd.Timestamp(date).strftime("%Y-%m-%d") for date in extended_dates])
#         ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
#         if "Halmstad" in station:
#             ax.yaxis.set_major_locator(mdates.DayLocator(interval=5))
#         elif "Umeå" in station:
#             # ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
#             # ax.set_xticks(range(0, 23, 3))
#             # ax.xaxis.set_major_locator(mdates.HourLocator(interval=10))
#             ax.yaxis.set_major_locator(mdates.DayLocator(interval=7))
           
#         elif "Upsala" in station:
#             ax.yaxis.set_major_locator(mdates.DayLocator(interval=1))
#             ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

#         # Check if the station is Umeå and apply custom x-ticks
#         if "Umeå" in station:
#             ax.set_xticks([0, 3, 7, 11, 15, 19, 23])  # Only for Umeå plot, set x-ticks at 00:00, 12:00, 23:00
#             ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))  # Display hour only on x-axis
#             ax.set_xticklabels(['0', '3', '7', '11', '15', '19', '23'])
#         # Customize plot appearance
#         plt.title(f"Missing Data for {station}")

#         plt.ylabel("Date")
#         plt.grid(visible=True, which="minor", linestyle="--", alpha=0.5)
#         plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12, title="Stations", frameon=True, edgecolor='black', facecolor='lightgray')

#         # Show plot
#         plt.tight_layout()
#         output_dir = ".grapfic_decription"
#         os.makedirs(output_dir, exist_ok=True)
#         plt.savefig(f"{output_dir}/{station}missing_data_SCATTER.png")
#         plt.show()

# # Call the function
# plot_missing_data_with_custom_axis()


# # -- Graphics Level 1: Plot DataFrame Time Series --

# # Plot each station's data over time
# plt.figure(figsize=(12, 6))
# for column in df.columns:
#     plt.plot(df.index, df[column], label=column)

# plt.title("Temperature Over Time by Station", fontsize=16)
# plt.xlabel("Timestamp", fontsize=12)
# plt.ylabel("Temperature (°C)", fontsize=12)
# plt.legend(title="Station", loc="upper left")
# plt.grid()
# plt.show()

# # -- Graphics Level 2: Visualize Statistics --

# # Transpose stats for easier plotting
# stats_transposed = stats.T

# # Barplot of summary statistics (mean, std, min, max)
# plt.figure(figsize=(12, 6))
# stats_transposed[['mean', 'std', 'min', 'max']].plot(kind='bar', figsize=(12, 6))
# plt.title("Summary Statistics for Each Station", fontsize=16)
# plt.xlabel("Station", fontsize=12)
# plt.ylabel("Temperature (°C)", fontsize=12)
# plt.grid()
# plt.show()

# # Plot temperature frequency distribution for each station
# plt.figure(figsize=(15, 10))
# for column in df.columns:
#     sns.histplot(df[column], bins=15, kde=True, label=column)

# plt.title("Temperature Frequency Distribution by Station", fontsize=16)
# plt.xlabel("Temperature (°C)", fontsize=12)
# plt.ylabel("Frequency", fontsize=12)
# plt.legend(title="Station", loc="upper right")
# plt.grid()
# plt.show()

# # Plot separate histograms for each station
# for column in df.columns:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df[column], bins=15, kde=True, color='blue')
#     plt.title(f"Temperature Frequency Distribution - {column}", fontsize=16)
#     plt.xlabel("Temperature (°C)", fontsize=12)
#     plt.ylabel("Frequency", fontsize=12)
#     plt.grid()
#     plt.show()



# # Beräkna korrelationen
# correlation_matrix = df.corr()

# # Visualisera med en heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap=custom_cmap, cbar=True, fmt=".2f")
# plt.title("Korrelation mellan temperaturdata på olika stationer", fontsize=14)
# plt.show()

if __name__=="__main__":
    param = [[1, "TEMPERATUR", "°C"], [6, "LUFTFUKTIGHET", "%"]]
    for p in param:
        # hämta sparade data 
        data_temp = p_d.data_from_file(param=p[0])
        # ta fram sista 3 dagara
        three_days = p_d.extract_for_statistics(data=data_temp)
        df, stats = p_d.data_describe(three_days)

        missing_data(df)