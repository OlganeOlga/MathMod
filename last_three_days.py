import requests
import os
import json
from datetime import datetime, timedelta, timezone
import pytz

OUTPUT_DIR = "smhi_three_days"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_smhi_data(url, save_path):
    """
    Downloads data from a given SMHI API URL and saves it to a file.

    Args:
        url (str): API endpoint URL.
        save_path (str): Path to save the JSON file.

    Returns:
        bool: True if data was saved successfully, False otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download data: {e}")
        return False


def get_data(param="1", time="latest-months"):
    """
    Fetch data from specified SMHI stations and save to JSON files.

    Args:
        param (str): SMHI parameter ID (e.g., "1" for temperature, "6" for humidity).
        time (str): Time period for the data. Defaults to "latest-months".

    Returns:
        dict: Dictionary mapping station names to their SMHI IDs.
    """
    stations_url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param}/station.json"
    to_get = ["Uppsala Flygplats", "Halmstad flygplats", "Umeå Flygplats"]
    existed_stations = {}

    try:
        response = requests.get(stations_url)
        response.raise_for_status()
        stations = response.json()

        for station in stations.get("station", []):
            if station["name"] in to_get:
                id = station["id"]
                data_url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param}/station/{id}/period/{time}/data.json"
                save_path = os.path.join(OUTPUT_DIR, f"{id}_{param}.json")
                if download_smhi_data(data_url, save_path):
                    existed_stations[station["name"]] = id
        return existed_stations

    except requests.exceptions.RequestException as e:
        print(f"From get_data: Error fetching station list: {e}")
        return {}


def filter_last_72_hours(file_path):
    """
    Filters timestamps and values for the last 72 hours from the downloaded JSON data.

    Args:
        file_path (str): Path to the JSON file containing SMHI station data.

    Returns:
        list: Filtered data entries (list of dictionaries) from the last 72 hours.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        # Current time in Sweden timezone
        current_time = datetime.now(pytz.timezone("Europe/Stockholm"))
        cutoff_time = current_time - timedelta(hours=72)

        # Filter timestamps
        filtered_data = [
            entry for entry in data["value"]
            if datetime.fromtimestamp(entry["date"] / 1000, tz=pytz.timezone("Europe/Stockholm")) > cutoff_time
        ]

        return filtered_data

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"From Filter_last_72_hours: Error processing file: {e}")
        return []


def get_last_72_hours_data(param="1"):
    """
    Fetch data for selected stations and filter it for the last 72 hours.

    Args:
        param (str): SMHI parameter ID (e.g., "1" for temperature, "6" for humidity).

    Returns:
        dict: Filtered data for each station (station name mapped to filtered data).
    """
    stations = get_data(param, time="latest-months")
    filtered_results = {}

    for station_name, station_id in stations.items():
        file_path = os.path.join(OUTPUT_DIR, f"{station_id}_{param}.json")
        filtered_data = filter_last_72_hours(file_path)
        filtered_results[station_name] = filtered_data

    return filtered_results


# get results for needed param
if __name__ == "__main__":
    param_id = ["1", "6"]  # Change parameter ID as needed (e.g., 6 for humidity)
    results = []
    for i in range(len(param_id)):
        results.append(get_last_72_hours_data(param=param_id[i]))

    for station, data in results[0].items():
        print(f"\nStation: {station}")
        for entry in data:
            timestamp = datetime.fromtimestamp(entry["date"] / 1000, tz=pytz.timezone("Europe/Stockholm"))
            value = entry["value"]
            print(f"Time: {timestamp}, Value: {value}")
