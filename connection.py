import requests

# Define the API URL
url = "https://api2.miktex.org/hello"

try:
    # Make a GET request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        print("Success:", response.text)
    else:
        print("Error:", response.status_code, response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)