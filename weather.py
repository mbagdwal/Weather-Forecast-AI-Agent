#call weather API
import requests;

def get_coordinates(location):
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        
        params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if "results" not in data or not data["results"]:
            raise ValueError(f"Location not found: {location}")

        result = data["results"][0]

        return {
            "name": result["name"],
            "country": result.get("country"),
            "latitude": result["latitude"],
            "longitude": result["longitude"]
        }
    except requests.exceptions.HTTPError:
        print("Error: Server taking too long to get coordinates. Please try after sometime.")


def get_forecast(location, days=3):
    
    coordinates = get_coordinates(location);

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "forecast_days": days,
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=10)

    response.raise_for_status();

    return {
        "location": coordinates["name"],
        "country": coordinates["country"],
        "forecast": response.json()["daily"]
    }


def get_weather(location):

    try:
        coordinates = get_coordinates(location)

        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": coordinates["latitude"],
            "longitude": coordinates["longitude"],
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m"
            ],
            "timezone": "auto"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        return {
            "location": coordinates["name"],
            "country": coordinates["country"],
            "temperature_c": data["current"]["temperature_2m"],
            "humidity": data["current"]["relative_humidity_2m"],
            "feels_like_c": data["current"]["apparent_temperature"],
            "precipitation_mm": data["current"]["precipitation"],
            "weather_code": data["current"]["weather_code"],
            "wind_speed_kmh": data["current"]["wind_speed_10m"]
        }
    except requests.exceptions.HTTPError:
            print("Error: Server taking too long to weather. Please try after sometime.")

# print(get_forecast('Dubai'));