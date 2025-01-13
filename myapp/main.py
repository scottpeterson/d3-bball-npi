# src/myapp/main.py
from pathlib import Path
import time
from .load_teams import load_teams
from .process_games_iteration import process_games_iteration
from .load_games import load_games
from .save_npi_results_to_csv import save_npi_results_to_csv
from .process_games_bidirectional import process_games_bidirectional
from .simulation import simulate_game, predict_and_simulate_game


def main():
    """Main entry point for the application."""
    data_path = Path(__file__).parent / "data"
    year = "2025"
    NUM_ITERATIONS = 50
    try:
        valid_teams = load_teams(data_path, year)
        games = load_games(data_path, year, valid_teams)
        print(f"Total number of loaded games: {len(games)}")

        start_total_time = time.time()
        previous_iteration_npis = None
        final_teams = None

        for i in range(NUM_ITERATIONS):
            iteration_number = i + 1

            # Calculate opponent NPIs for this iteration
            if iteration_number == 1:
                opponent_npis = {team_id: 50 for team_id in valid_teams}
            else:
                opponent_npis = {
                    team_id: previous_iteration_npis[team_id]
                    for team_id in valid_teams
                    if team_id in previous_iteration_npis
                }

            # Handle any teams that don't have a previous NPI
            for team_id in valid_teams:
                if team_id not in opponent_npis:
                    opponent_npis[team_id] = 50  # Default to 50 if no previous NPI

            teams = process_games_iteration(
                games, valid_teams, previous_iteration_npis, iteration_number
            )

            if iteration_number == NUM_ITERATIONS:
                final_teams = teams
                save_npi_results_to_csv(teams)

            previous_iteration_npis = {
                team_id: stats["npi"]
                for team_id, stats in teams.items()
                if stats["has_games"]
            }

        total_time = time.time() - start_total_time
        print(f"\nTotal processing time: {total_time:.3f} seconds")
        print(f"Average time per iteration: {total_time/NUM_ITERATIONS:.3f} seconds")

        # Get the number of counted games from the final iteration
        total_games = 0
        for team_id, team_data in final_teams.items():
            total_games += len(team_data["all_game_npis"])

        print(f"Total number of games in the data: {len(games)}")
        print(f"Total number of games processed in the final iteration: {total_games}")

        return final_teams  # Add this return statement

    except Exception as e:
        print(f"Error processing: {e}")
        raise


if __name__ == "__main__":
    main()
