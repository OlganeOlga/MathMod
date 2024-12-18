import requests
import json
import os
import requests
# variables
TO_GET = ["Uppsala Flygplats", "Halmstad flygplats", "Umeå Flygplats"]
# Directory to save the data files
OUTPUT_DIR = "smhi_data_temp_fukt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PARAMS = [1, 6]

def download_smhi_data(url, save_path):
    """
    Dowloads data from a station 

    Args:
        url (_type_): _description_
        save_path (str, optional): _description_. Defaults to "smhi_data.json".

    Returns:
        _bool_: True if data existing otherwis False
    """
    try:
        # Fetch CSV data
        response = requests.get(url)
        response.raise_for_status()  # Check if the request succeeded
        
        result = json.loads(response.content)
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
        return True

    except requests.exceptions.RequestException as e:
        return False

def get_data(param="1", time="latest-months"):
    """
    Function find stations providing parameter and returns data from the stations
    Args:
        param (str, optional): parametr id. Defaults to "1".
    Returns: array[[str,str]]: array of statons provideing data [name, id].
    """
    # Define the URL for stations providing air temperature data
    stations_url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param}/station.json"
    # Fetch station data
    existed_stations = {}
    to_get = ["Uppsala Flygplats", "Halmstad flygplats", "Umeå Flygplats"]
    try:
        response = requests.get(stations_url)
        response.raise_for_status()  # Raise an error for failed requests

        # Parse JSON response
        stations = response.json()        
        for station in stations.get("station", []):
            if station["name"] in to_get:      
                # URL for fetching the latest-day data for the selected station
                id = station["id"]
                data_url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param}/station/{id}/period/{time}/data.json"
                save_path = f"{OUTPUT_DIR}/{id}_{param}.json"
                name = download_smhi_data(data_url, save_path)
                if name:
                    existed_stations[station["name"]] = id
        print(existed_stations)
        return existed_stations
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    for p in PARAMS:
        get_data(p)
