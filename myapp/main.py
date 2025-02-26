# src/myapp/main.py
import time
from pathlib import Path

from .load_games import load_games
from .load_teams import load_teams
from .process_games_iteration import process_games_iteration
from .save_npi_results_to_csv import save_npi_results_to_csv


def main(use_season_results=False):
    """Main entry point for the application."""
    data_path = Path(__file__).parent / "data"
    year = "2025"
    NUM_ITERATIONS = 30 if use_season_results else 80

    try:
        valid_teams = load_teams(data_path, year)
        games = load_games(data_path, year, valid_teams, use_season_results)
        print(f"Total number of loaded games: {len(games)}")

        start_total_time = time.time()
        # Initialize once
        opponent_npis = {team_id: 50 for team_id in valid_teams}
        final_teams = None
        total_games = 0

        for i in range(NUM_ITERATIONS):
            iteration_number = i + 1

            teams = process_games_iteration(
                games, valid_teams, opponent_npis, iteration_number
            )

            if iteration_number == NUM_ITERATIONS:
                final_teams = teams
                save_npi_results_to_csv(teams)

                # Calculate total games in final iteration
                for team_id, team_data in teams.items():
                    total_games += len(team_data["all_game_npis"])

            # Update in place for next iteration
            opponent_npis.clear()
            opponent_npis.update(
                {
                    team_id: stats["npi"]
                    for team_id, stats in teams.items()
                    if stats["has_games"]
                }
            )

        total_time = time.time() - start_total_time
        print(f"\nTotal processing time: {total_time:.3f} seconds")
        print(f"Average time per iteration: {total_time/NUM_ITERATIONS:.3f} seconds")
        print(f"Total number of games in the data: {len(games)}")
        print(f"Total number of games processed in the final iteration: {total_games}")

        return final_teams

    except Exception as e:
        print(f"Error processing: {e}")
        raise


if __name__ == "__main__":
    main()
