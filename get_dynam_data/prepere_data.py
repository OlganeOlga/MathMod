import json
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os
import pytz
"""
get data dinamicaly
"""

STATIONS = {'Halmstad flygplats': 62410, 'Umeå Flygplats': 140480, 'Uppsala Flygplats': 97530}
DIR = 'smhi_data_temp_fukt'
PARAMS_NAME = {1:"Temperatur", 6:"Luftfuktighet"}

def data_from_file(stations=STATIONS,
                   dir: str=DIR,
                   param: int =1,
                   hours: int = 73):
    """
    Get data form fails return dictionary with name : data
    Args:
        data (dictionary): name:id of stations

    Returns:
        _dictionary_: name: data of stations
    """
    current_time = datetime.now(pytz.timezone("Europe/Stockholm"))
    rounded_time = current_time.replace(minute=0, second=0, microsecond=0)
    print(rounded_time)
    cutoff_time = rounded_time - timedelta(hours=hours)

    station_data = {}
    try:
        for name, station_id in stations.items():
            file_path = os.path.join(dir, f"{station_id}_{param}.json")
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Filter and extract the last N hours data for the specific parameter
                filtered_data = [
                    entry for entry in data.get("value", [])
                    if datetime.fromtimestamp(entry["date"] / 1000, tz=pytz.timezone("Europe/Stockholm")) >= cutoff_time
                ]
            station_data[name] = filtered_data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")
    return station_data

def extract_for_statistics(data=data_from_file(param=1)):
    """
    Extracts data for easier statistisk process

    Args:
        data (_dict_): dictionary containing name_of_station: data from station
    Returns:
        _dict_: dictionary containing name_of_station:{ dictionary{timestamp:temp}}
    """
    new_data = {}
    for key, value in data.items():
        value_set = {}
        for item in value:
            value_set[item['date']] = item['value']
        new_data[key] = value_set
    return new_data

def data_describe(data=extract_for_statistics()):
    """Describes data

    Args:
        data (_dict_): dictionary containing name_of_station:{ dictionary{timestamp:temp}}
        Defaults to extract_for_statistics().

    Returns:
    tuple: (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)
        - First DataFrame: Contains the cleaned and preprocessed data ready for further analysis.
        - Second DataFrame: Descriptive statistics for each location, including count, mean, std, min, and quartile values.
    """
    # Convert the dictionary into a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='columns')
        
    # Convert all values to numeric
    df = df.apply(pd.to_numeric)
    # Convert the index (timestamp) to a readable datetime format
    df.index = pd.to_datetime(df.index, unit='ms')

    # Format the index as a readable string (optional)
    df.index = df.index.strftime('%Y-%m-%d %H:%M:%S')

    # Generate statistics for each location
    stats = df.describe()

    # Return the data and the statistics
    return df, stats

def append_to_markdown(df: pd.DataFrame, filename: str = 'project/RAPPORT.md'):
    """
    Append the DataFrame as a Markdown table to an existing Markdown file.
    
    Args:
        df (pd.DataFrame): DataFrame to append as a Markdown table.
        filename (str): The name of the existing Markdown file.
    NO returns
    """
    # Convert DataFrame to Markdown format
    markdown_table = df.to_markdown()

    # Read the existing contents of the Markdown file
    try:
        with open(filename, 'a', encoding='utf-8') as file:  # Open in append mode
            file.write("\n")  # Add a newline before appending
            file.write("## Data Table\n")  # Optional header for the table
            file.write(markdown_table)  # Append the markdown table to the file
            print(f"Markdown table appended to {filename}")
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
   
    
if __name__ == "__main__":
    data = data_from_file(param=1)
    three_days = extract_for_statistics(data=data)
    df, stats = data_describe(three_days)
    print(stats)
    append_to_markdown(df=stats)
    data = data_from_file(param=6)
    three_days = extract_for_statistics(data=data)
    df, stats = data_describe(three_days)
    print(stats)
    append_to_markdown(df=stats)
