import requests
from pathlib import Path


def games_getter(url, year="2025"):
    output_path = Path(__file__).parent / "data" / year
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "games.txt"

    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_file, "w") as f:
            f.write(response.text)
        return True
    except (requests.RequestException, IOError) as e:
        print(f"Error: {e}")
        return False
