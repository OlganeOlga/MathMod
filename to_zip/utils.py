import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import scipy as sci
import seaborn as sns


STATIONS = {'Halmstad flygplats': 62410, 'Uppsala Flygplats': 97530, 'Umeå Flygplats': 140480}

def change_to_markdown(df: pd.DataFrame, units="°C"):
    """
    Append the DataFrame as a Markdown table to an existing Markdown file.
    
    Args:
        df (pd.DataFrame): DataFrame to append as a Markdown table.
        filename (str): The name of the existing Markdown file.
    NO returns
    """
    # rename columns: add units
    if units is not None:
        df.rename(columns={col: f"{col}({units})" for col in df.columns}, inplace=True)
    
    # Convert DataFrame to Markdown format
    markdown_table = df.to_markdown()
    return markdown_table

def append_to_markdown(data_frame, filename: str = 'RAPPORT.md', header=""):
    md_str = data_frame.to_markdown()
    # Read the existing contents of the Markdown file
    try:
        with open(filename, 'a', encoding='utf-8') as file:  # Open in append mode
            file.write("\n")  # Add a newline before appending
            file.write(f'### {header}\n')  #  header for the table
            file.write(md_str)  # Append the markdown table to the file
            print(f"Markdown table appended to {filename}")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")

def save_to_mdfile(data_frame, filename: str = 'RAPPORT.md', dir_name: str = "describe"):
    markdown_table = data_frame.to_markdown()
    # Specify the file path where you want to save the markdown file
    file_path = f'{dir_name}/{filename}'

    # Write the markdown table to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(markdown_table)

def table_to_picture(table: pd.DataFrame, saving_path: str):
    # Create a plot to render the table as an image
    fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the size of the image

    # Hide the axes
    ax.xaxis.set_visible(False)  # Hide x-axis
    ax.yaxis.set_visible(False)  # Hide y-axis
    ax.set_frame_on(False)  # No frame

    # Render the table
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc='center', cellLoc='center')

    # Step 5: Save the figure as an image (PNG)
    plt.savefig(saving_path, bbox_inches='tight', dpi=300)
    plt.close()

def stat_norm_distribution(df: pd.DataFrame):
    """
    Plots the distribution (frequency) of temperatures across multiple stations.
    The x-axis represents the temperature values, and the y-axis represents the frequency of each temperature.
    """
    data= df.to_dict(orient="list")
    result = {}
    for key, value in data.items():
        stat, p_value = sci.stats.shapiro(value)
        result[key] = [stat, p_value]
    return result

def define_axix(axes, x_lable, y_lable, array1, array2=None):
    for ax, (key, values) in zip(axes, array1.items()):
        sns.histplot(values, kde=True, bins=15, color='blue', edgecolor='black', ax=ax)
        
        # Add titles and labels
        ax.set_title(f'Frekvens spridning i {key}', fontsize=10)
        ax.set_xlabel(x_lable, fontsize=12)
        ax.set_ylabel(y_lable, fontsize=12)
        
        # Add Shapiro-Wilk test results as annotation
        stat, p_value = array2[key]
        ax.text(0.05, 0.95, f"Shapiro-Wilk:\nStat: {stat:.4f}\nP-value: {p_value:.4g}",
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

def botstrapping(date_frame, station_name, parameter):
    filtered_data = date_frame[(date_frame["station_name"] == station_name) & (date_frame["parameter"] == parameter)]
    f_data = filtered_data['value'].tolist()
    f_mean = sum(f_data)/len(f_data)
    print(f"standart mean: {f_mean}")
    # För att lagra medelvärden för varje bootstrap-prov
    bootstrap_means = []
    # Antal bootstrap-prover
    n_iterations = 10000
    # Slumpmässiga stickprov med återläggning
    for _ in range(n_iterations):
        # Skapa ett stickprov med återläggning
        sample = np.random.choice(f_data, size=len(f_data), replace=True)
        # Beräkna medelvärdet för detta stickprov och spara det
        bootstrap_means.append(np.mean(sample))

    # Beräkna det genomsnittliga medelvärdet från alla stickprov
    estimated_mean = np.mean(bootstrap_means)

    # Visa resultatet
    print(f"Uppskattat väntevärde (med bootstrapping): {estimated_mean}")
