import json
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os
import pytz

# Station ID mapping
DEFAULT_STATIONS = {
    'Halmstad flygplats': 62410,
    'Umeå Flygplats': 140480,
    'Uppsala Flygplats': 97530
}
DEFAULT_DIR = 'smhi_data_temp_fukt'
DEFOULT_PARAM = [1, 6]
PARAMS_NAME = {1:"Temperatur", 6:"Luftfuktighet"}
def filter_and_extract_data(
    stations: Dict[str, int] = DEFAULT_STATIONS,
    directory: str = DEFAULT_DIR,
    params: List[int] = [1, 6],  # Handle multiple parameters (1 for temperature, 6 for humidity)
    hours: int = 73
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """
    Reads JSON files for each station and filters the data for the last N hours,
    then returns the data for multiple parameters in a simplified format for statistical processing.

    Args:
        stations (dict): A dictionary mapping station names to their IDs.
        directory (str): Path to the directory containing JSON files.
        params (list): A list of parameter IDs (e.g., [1, 6]).
        hours (int): The number of hours to filter the data. Defaults to 72.

    Returns:
        Dict[str, Dict[int, Dict[int, float]]]: A dictionary where:
            - keys are station names,
            - values are dictionaries with timestamps as keys,
            - each timestamp contains a dictionary with parameter values.
    """
    current_time = datetime.now(pytz.timezone("Europe/Stockholm"))
    rounded_time = current_time.replace(minute=0, second=0, microsecond=0)
    cutoff_time = rounded_time - timedelta(hours=hours)

    station_data = {}
    for name, station_id in stations.items():
        station_data[name] = {}
        for param in params:
            file_path = os.path.join(directory, f"{station_id}_{param}.json")
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                # Filter and extract the last N hours data for the specific parameter
                filtered_data = [
                    entry for entry in data.get("value", [])
                    if datetime.fromtimestamp(entry["date"] / 1000, tz=pytz.timezone("Europe/Stockholm")) > cutoff_time
                ]
                # Simplify the data for statistical processing
                for entry in filtered_data:
                    timestamp = entry["date"]
                    if timestamp not in station_data[name]:
                        station_data[name][timestamp] = {}
                    station_data[name][timestamp][param] = entry["value"]
            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found for station '{name}' and param '{param}'.")
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON file '{file_path}' for station '{name}' and param '{param}'.")

    return station_data

def dict_to_dataframes(data: Dict[str, Dict[int, Dict[int, float]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a nested dictionary of station data into two pandas DataFrames:
    one for 'Temperatur' and another for 'Luftfuktighet'.

    Args:
        data (dict): A dictionary with station names, timestamps, and parameter values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - First DataFrame contains temperature values ('Temperatur').
            - Second DataFrame contains humidity values ('Luftfuktighet').
    """
    rows = []  # List to store flattened rows

    for station_name, timestamps in data.items():
        for timestamp, params in timestamps.items():
            # Build each row as a flat dictionary
            row = {"station": station_name, "timestamp": timestamp}
            for param, value in params.items():
                name = PARAMS_NAME[param]  # Convert parameter index to its name
                row[f"{name}"] = value
            rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    # Convert the timestamp column to a readable datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Sort by station and timestamp for better readability
    df.sort_values(by=["station", "timestamp"], inplace=True)

    # Split the DataFrame into two: one for 'Temperatur' and one for 'Luftfuktighet'
    temperatur_df = df[["station", "timestamp", "Temperatur"]].copy()
    luftfuktighet_df = df[["station", "timestamp", "Luftfuktighet"]].copy()

    return temperatur_df, luftfuktighet_df

def dict_to_dataframe(data: Dict[str, Dict[int, Dict[int, float]]]) -> pd.DataFrame:
    """
    Converts a nested dictionary of station data into a pandas DataFrame.

    Args:
        data (dict): A dictionary with station names, timestamps, and parameter values.

    Returns:
        pd.DataFrame: A DataFrame where each row contains station, timestamp, and parameter values.
    """
    rows = []  # List to store flattened rows

    for station_name, timestamps in data.items():
        for timestamp, params in timestamps.items():
            # Build each row as a flat dictionary
            row = {"station": station_name, "timestamp": timestamp}
            for param, value in params.items():
                name = PARAMS_NAME[param]
                row[f"{name}"] = value
            rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    # Convert the timestamp column to a readable datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')

    # Sort by station and timestamp for better readability
    df.sort_values(by=["station", "timestamp"], inplace=True)

    return df

def dataframe_to_pivot(df):
    """
    Converts a DataFrame into a pivot table aligned by timestamp and station.

    Args:
        df (pd.DataFrame): Input DataFrame with station, timestamp, and parameter values.

    Returns:
        pd.DataFrame: Pivot table with aligned timestamps and parameter values.
    """
    # Step 1: Ensure timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    # Step 2: Convert 'Temperatur' and 'Luftfuktighet' to numeric (handle strings)
    df["Temperatur"] = pd.to_numeric(df["Temperatur"], errors="coerce")
    df["Luftfuktighet"] = pd.to_numeric(df["Luftfuktighet"], errors="coerce")

    # Step 3: Pivot table to align timestamps and stations
    pivot_df = df.pivot_table(index="timestamp", columns="station", values=["Temperatur", "Luftfuktighet"])

    # Step 4: Flatten multi-index columns for readability
    pivot_df.columns = [f"{param}_{station}" for param, station in pivot_df.columns]

    # Step 5: Sort the table by timestamp
    pivot_df = pivot_df.sort_index()

    return pivot_df


if __name__ == "__main__":
    data = filter_and_extract_data()
    temp, fukt = dict_to_dataframes(data)
    print(temp)
