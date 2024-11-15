import sys
from pathlib import Path
import csv
from .main import load_teams, process_games_bidirectional
from .games_getter import games_getter


def run_bidirectional():
    if len(sys.argv) > 2:
        year = sys.argv[2]
    else:
        year = "2025"

    print(f"Processing data for year {year}")

    base_path = Path(__file__).parent / "data"

    valid_teams = load_teams(base_path, year)
    games_path = base_path / year / "games.txt"
    results = process_games_bidirectional(games_path, valid_teams)

    if not results:
        print("No valid results to write")
        return

    newest_date = ""
    for result in results:
        date = result.split(",")[0]
        if date > newest_date:
            newest_date = date

    month, day, year_part = newest_date.split("/")
    filename_date = f"{month}{day}{year_part}"

    output_filename = f"{filename_date}_WBB_results.csv"
    output_path = base_path / year / output_filename

    (base_path / year).mkdir(exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        csvfile.write("Date,Team A,Team B,Home/Away,Result\n")
        for result in results:
            csvfile.write(f"{result}\n")

    print(f"Results written to: {output_filename}")
    print(f"Total games processed: {len(results)//2}")


def run_games_getter():
    url = (
        "https://masseyratings.com/scores.php?s=604303&sub=11620&all=1&mode=2&format=1"
    )
    if games_getter(url, "2025"):
        print("Successfully saved webpage data to games.txt")


def run_main():
    from .main import main

    main()


if __name__ == "__main__":
    import sys

    commands = {
        "bidirectional": run_bidirectional,
        "main": run_main,
        "get_games": run_games_getter,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Available commands:")
        for cmd in commands:
            print(f"  - {cmd}")
        print("\nUsage:")
        print(
            "  bidirectional [year]  - Process games for specified year (default: 2024)"
        )
        sys.exit(1)

    commands[sys.argv[1]]()
