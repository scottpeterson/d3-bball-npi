import sys
from pathlib import Path
import csv
from .main import load_teams, process_games_bidirectional, predict_and_simulate_game
from .games_getter import games_getter
from .simulation import load_efficiency_data, simulate_full_season
from .multi_season_simulator import save_simulation_stats
from .conf_tournaments import load_conference_data
from .eff_getter import efficiency_getter
from .massey_ratings_getter import massey_ratings_getter
from .team_id_getter import team_ids_getter

def run_team_ids_getter():
    url = "https://masseyratings.com/scores.php?s=604303&sub=11620&all=1&mode=3&sch=on&format=2"
    team_ids_getter(url)


def run_massey_ratings_getter():
    url = "https://masseyratings.com/cbw2025/ncaa-d3/ratings"
    massey_ratings_getter(url)


def run_efficiency_getter():
    url = "https://d3datacast.com/efficiency-ratings/wbb-efficiency-ratings/"
    efficiency_getter(url)


def run_multiple_simulations_command():
    """Run multiple season simulations and generate statistics."""
    year = "2025"
    base_path = Path(__file__).parent / "data"
    NUM_SIMULATIONS = 100

    try:
        print(f"\nStarting {NUM_SIMULATIONS} simulations...")
        print("-" * 50)
        
        # Import necessary functions
        from .multi_season_simulator import (
            run_multiple_simulations,
            save_simulation_stats,
            load_conference_data,
        )

        # Load team mappings
        teams_mapping = {}
        mapping_path = base_path / year / "teams_mapping.txt"
        with open(mapping_path, "r") as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    team_id = parts[0].strip()
                    scott_name = parts[2].strip()
                    teams_mapping[team_id] = scott_name

        # Load conference data
        conference_teams = load_conference_data(base_path, year)

        # Run simulations - now includes tournament_stats
        stats, bid_thief_counts, per_sim_bid_thieves, _ = run_multiple_simulations(
            base_path, year, NUM_SIMULATIONS
        )

        # Save regular simulation stats (now includes tournament stats)
        save_simulation_stats(stats, base_path, year, conference_teams)

        # Save bid thief stats
        bid_thief_path = base_path / year / "bid_thieves.csv"
        with open(bid_thief_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Team", "Bid Thief Count", "Percentage"])
            for team_id, count in sorted(bid_thief_counts.items(), key=lambda x: x[1], reverse=True):
                team_name = teams_mapping.get(team_id, team_id)  # Fallback to ID if mapping not found
                percentage = (count / NUM_SIMULATIONS) * 100
                writer.writerow([team_name, count, f"{percentage:.1f}%"])
        print("Bid thief statistics have been saved to bid_thieves.csv")

        # Save per-simulation bid thief details
        bid_thief_details_path = base_path / year / "bid_thieves_per_sim.csv"
        with open(bid_thief_details_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Simulation", "Bid Thief Count"])
            for sim_num, count in enumerate(per_sim_bid_thieves, 1):
                writer.writerow([sim_num, count])

        print("\nSimulation statistics have been saved to simulation_stats.csv")

    except Exception as e:
        print(f"Error in simulations: {e}")


def run_simulate_season():
    """
    Simulate all remaining games in the season including conference tournaments
    """
    year = "2025"
    base_path = Path(__file__).parent / "data"

    try:
        # Load teams first
        valid_teams = load_teams(base_path, year)

        # Load efficiency data
        team_data = load_efficiency_data(base_path, int(year))

        # Run full season simulation
        if simulate_full_season(base_path, year, valid_teams, team_data):
            print("\nFull season simulation completed successfully")
        else:
            print("\nFull season simulation failed")

    except Exception as e:
        print(f"Error in season simulation: {e}")


def run_predict_game():
    # Hardcoded values
    team_a_id = "325"
    team_b_id = "160"
    year = "2025"

    base_path = Path(__file__).parent / "data"

    try:
        # Load teams first to validate IDs
        valid_teams = load_teams(base_path, year)
        if team_a_id not in valid_teams or team_b_id not in valid_teams:
            print(f"Error: Invalid team ID(s)")
            print(
                f"Team A ({team_a_id}): {'Found' if team_a_id in valid_teams else 'Not found'}"
            )
            print(
                f"Team B ({team_b_id}): {'Found' if team_b_id in valid_teams else 'Not found'}"
            )
            return

        probabilities, result = predict_and_simulate_game(
            base_path, team_a_id, team_b_id, int(year)
        )

        # Print predictions
        print("\nPredicted Win Probabilities:")
        print(f"{'Team':<30} {'Win Probability':<15}")
        print("-" * 45)
        for team_id, prob in probabilities.items():
            team_name = valid_teams[team_id]
            print(f"{team_name:<30} {prob:>6.1%}")

        # Print simulation result
        print("\nSimulated Game Result:")
        print("-" * 45)
        winner_name = valid_teams[result.winner_id]
        loser_name = valid_teams[result.loser_id]
        print(
            f"{winner_name} {result.winning_score}, {loser_name} {result.losing_score}"
        )
        if result.was_upset:
            print("UPSET!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_bidirectional():
    if len(sys.argv) > 2:
        year = sys.argv[2]
    else:
        year = "2025"

    print(f"Processing data for year {year}")

    base_path = Path(__file__).parent / "data"

    valid_teams = load_teams(base_path, year)
    games_path = base_path / year / "games.txt"
    # games_path = base_path / year / "season_results.txt"
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
    url = "https://masseyratings.com/scores.php?s=604303&sub=11620&all=1&mode=2&sch=on&format=1"
    if games_getter(url, "2025"):
        print("Successfully saved webpage data to games.txt")


def run_main():
    from .main import main

    main()


if __name__ == "__main__":
    commands = {
        "bidirectional": run_bidirectional,
        "main": run_main,
        "get_games": run_games_getter,
        "predict_game": run_predict_game,
        "simulate_season": run_simulate_season,
        "run_multiple": run_multiple_simulations_command,
        "eff": run_efficiency_getter,
        "massey": run_massey_ratings_getter,
        "teams": run_team_ids_getter
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Available commands:")
        for cmd in commands:
            print(f" - {cmd}")
        print("\nUsage:")
        print(
            " bidirectional [year] - Process games for specified year (default: 2024)"
        )
        print(" predict_game - Predict game outcome for hardcoded teams")
        print(" simulate_season - Simulate all remaining games in the season")
        print(" simulate multiple seasons")
        print("run efficiency getter")
        print("run massey ratings getter")
        print("run team ids getter")
        sys.exit(1)

    commands[sys.argv[1]]()
