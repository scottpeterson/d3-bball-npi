import requests
from pathlib import Path
import pandas as pd
import re
from io import StringIO


def efficiency_getter(url, year="2025"):
    """
    Scrapes efficiency ratings data, applies team mappings, and formats output
    """
    output_path = Path(__file__).parent / "data" / year
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "eff.txt"

    snyder_mapping_file = output_path / "snyder_team_mapping.txt"
    team_mapping_file = output_path / "teams_mapping.txt"

    try:
        if not snyder_mapping_file.exists() or not team_mapping_file.exists():
            raise ValueError("Required mapping files not found")

        snyder_mappings = pd.read_csv(snyder_mapping_file)
        team_mappings = pd.read_csv(team_mapping_file, skipinitialspace=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        sheet_url_match = re.search(
            r'src="(https://docs\.google\.com/spreadsheets/d/e/[^"]+)"', response.text
        )
        if not sheet_url_match:
            raise ValueError("Could not find Google Sheets URL in page")

        sheet_url = sheet_url_match.group(1)
        export_url = sheet_url.replace("/pubhtml?", "/pub?output=csv&")

        sheet_response = requests.get(export_url, headers=headers)
        sheet_response.raise_for_status()

        column_names = [
            "Rank",
            "Team",
            "Conference",
            "Record",
            "AdjEM",
            "AdjO",
            "ORank",
            "AdjD",
            "DRank",
            "AdjT",
            "TRank",
            "SOS1",
            "SRank1",
            "SOS2",
            "SRank2",
            "SOS3",
            "SRank3",
            "SOS4",
            "SRank4",
            "SOS5",
            "SRank5",
            "NCSOS",
            "NCRank",
        ]

        df = pd.read_csv(
            StringIO(sheet_response.text),
            skiprows=7,
            names=column_names,
            skipinitialspace=True,
        )

        # Filter out header rows
        df = df[
            ~df["Team"].str.contains("Nat'l", na=True)
            & ~df["Team"].str.contains("Team", na=True)
            & df["Team"].notna()
            & df["Rank"].astype(str).str.match(r"^\d+$")
        ]

        # Apply Snyder name mappings
        snyder_map = dict(
            zip(
                snyder_mappings["SNYDER NAME"].str.strip(),
                snyder_mappings["NAME NEEDED"].str.strip(),
            )
        )
        df["Team"] = df["Team"].replace(snyder_map).str.strip()

        # Convert team names to uppercase for case-insensitive matching
        df["Team"] = df["Team"].str.upper()
        massey_map = dict(
            zip(
                team_mappings["Scott_Name"].str.strip().str.upper(),
                team_mappings["Massey_id"],
            )
        )

        df["teamid"] = df["Team"].map(massey_map)

        # Drop any rows where we couldn't find a Massey ID
        df = df.dropna(subset=["teamid"])
        df["teamid"] = df["teamid"].astype(int)

        df = df[["teamid", "AdjEM", "AdjT"]]
        df.columns = ["teamid", "adjEm", "adjT"]
        df["adjEm"] = df["adjEm"].str.replace("+", "").astype(float)
        df = df.sort_values("teamid")

        df.to_csv(output_file, index=False)
        print(f"Successfully saved data with {len(df)} rows")
        return True

    except (requests.RequestException, IOError, ValueError) as e:
        print(f"Error: {e}")
        return False
