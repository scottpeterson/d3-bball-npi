# src/myapp/main.py
from collections import defaultdict
from pathlib import Path
import time
import csv


def load_teams(base_path, year):
    teams = {}
    year_str = str(year)
    teams_path = base_path / year_str / "teams.txt"
    mapping_path = base_path / year_str / "teams_mapping.txt"

    try:
        with open(teams_path, "r") as file:
            for line in file:
                try:
                    team_id, team_name = line.strip().split(",", 1)
                    teams[team_id.strip()] = team_name.strip()
                except Exception as e:
                    continue
        print(f"Loaded {len(teams)} teams from {year_str}/teams.txt")
    except FileNotFoundError:
        print(f"Warning: Could not find teams file at {teams_path}")
        return teams

    mapping_count = 0
    try:
        with open(mapping_path, "r") as file:
            for line in file:
                try:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        team_id = parts[0].strip()
                        mapped_name = parts[2].strip()
                        if team_id in teams:
                            teams[team_id] = mapped_name
                            mapping_count += 1
                except Exception as e:
                    continue
        print(
            f"Applied {mapping_count} name mappings from {year_str}/teams_mapping.txt"
        )
    except FileNotFoundError:
        print(f"Warning: Could not find mapping file at {mapping_path}")

    return teams


def calculate_game_npi(won, opponent_npi):
    win_component = 100 if won else 0
    base_npi = (win_component * 0.20) + (opponent_npi * 0.80)

    quality_bonus = max(0, (opponent_npi - 54.00) * (2 / 3)) if won else 0

    total_npi = base_npi + quality_bonus
    return total_npi


def calculate_initial_npi(wins, losses, ties, owp):
    total_games = wins + losses + ties
    winning_percentage = (wins / total_games * 100) if total_games > 0 else 0
    return (0.20 * winning_percentage) + (0.80 * owp)


def calculate_owp(games, valid_teams):
    records = defaultdict(lambda: {"wins": 0, "losses": 0, "games": 0})

    for game in games:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]
        team1_score = game["team1_score"]
        team2_score = game["team2_score"]

        if (
            team1_id not in valid_teams
            or team2_id not in valid_teams
            or (team1_score == 0 and team2_score == 0)
        ):
            continue

        records[team1_id]["games"] += 1
        records[team2_id]["games"] += 1

        if team1_score > team2_score:
            records[team1_id]["wins"] += 1
            records[team2_id]["losses"] += 1
        elif team2_score > team1_score:
            records[team2_id]["wins"] += 1
            records[team1_id]["losses"] += 1

    owp = {}
    for team_id in valid_teams:
        opponents_total_wins = 0
        opponents_total_losses = 0

        for game in games:
            if game["team1_score"] == 0 and game["team2_score"] == 0:
                continue

            if game["team1_id"] == team_id and game["team2_id"] in records:
                opp_record = records[game["team2_id"]]
                opp_wins = opp_record["wins"]
                opp_losses = opp_record["losses"]

                if game["team1_score"] > game["team2_score"]:
                    opp_losses -= 1
                elif game["team2_score"] > game["team1_score"]:
                    opp_wins -= 1

                opponents_total_wins += opp_wins
                opponents_total_losses += opp_losses

            elif game["team2_id"] == team_id and game["team1_id"] in records:
                opp_record = records[game["team1_id"]]
                opp_wins = opp_record["wins"]
                opp_losses = opp_record["losses"]

                if game["team2_score"] > game["team1_score"]:
                    opp_losses -= 1
                elif game["team1_score"] > game["team2_score"]:
                    opp_wins -= 1

                opponents_total_wins += opp_wins
                opponents_total_losses += opp_losses

        total_games = opponents_total_wins + opponents_total_losses
        owp[team_id] = (
            (opponents_total_wins / total_games * 100) if total_games > 0 else 50
        )

    return owp


def process_games_iteration(
    games, valid_teams, previous_iteration_npis=None, iteration_number=1
):
    owp = calculate_owp(games, valid_teams)
    records = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0})

    for game in games:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]
        team1_score = game["team1_score"]
        team2_score = game["team2_score"]

        if (
            team1_id not in valid_teams
            or team2_id not in valid_teams
            or (team1_score == 0 and team2_score == 0)
        ):
            continue

        if team1_score > team2_score:
            records[team1_id]["wins"] += 1
            records[team2_id]["losses"] += 1
        elif team2_score > team1_score:
            records[team2_id]["wins"] += 1
            records[team1_id]["losses"] += 1
        else:
            records[team1_id]["ties"] += 1
            records[team2_id]["ties"] += 1

    if iteration_number == 1:
        opponent_npis = {team_id: 50 for team_id in valid_teams}
    else:
        opponent_npis = {
            team_id: previous_iteration_npis[team_id]
            for team_id in valid_teams
            if team_id in previous_iteration_npis
        }
        for team_id in valid_teams:
            if team_id not in opponent_npis:
                opponent_npis[team_id] = owp[team_id]

    teams = defaultdict(
        lambda: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "npi": opponent_npis[team_id],
            "game_npis": [],
            "all_game_npis": [],
            "team_id": "",
            "team_name": "",
            "qualifying_wins": 0,
            "qualifying_losses": 0,
            "has_games": False,
        }
    )

    for team_id, team_name in valid_teams.items():
        teams[team_id]["team_id"] = team_id
        teams[team_id]["team_name"] = team_name

    # First pass: record basic stats
    for game in games:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]
        team1_score = game["team1_score"]
        team2_score = game["team2_score"]

        # Skip invalid teams and 0-0 games
        if (
            team1_id not in valid_teams
            or team2_id not in valid_teams
            or (team1_score == 0 and team2_score == 0)
        ):
            continue

        teams[team1_id]["has_games"] = True
        teams[team2_id]["has_games"] = True

        teams[team1_id]["games"] += 1
        teams[team2_id]["games"] += 1

        if team1_score > team2_score:
            teams[team1_id]["wins"] += 1
            teams[team2_id]["losses"] += 1
        elif team2_score > team1_score:
            teams[team2_id]["wins"] += 1
            teams[team1_id]["losses"] += 1
        else:
            teams[team1_id]["ties"] += 1
            teams[team2_id]["ties"] += 1

    # Second pass: calculate ALL potential game NPIs
    for game in games:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]
        team1_score = game["team1_score"]
        team2_score = game["team2_score"]

        if (
            team1_id not in valid_teams
            or team2_id not in valid_teams
            or (team1_score == 0 and team2_score == 0)
        ):
            continue

        team1_won = team1_score > team2_score
        team2_won = team2_score > team1_score

        team1_game_npi = calculate_game_npi(team1_won, opponent_npis[team2_id])
        team2_game_npi = calculate_game_npi(team2_won, opponent_npis[team1_id])

        teams[team1_id]["all_game_npis"].append((team1_game_npi, team1_won))
        teams[team2_id]["all_game_npis"].append((team2_game_npi, team2_won))

    # Third pass: filter and calculate final NPIs
    for team_id, team_data in teams.items():
        if not team_data["has_games"]:
            continue
        initial_npi = opponent_npis[team_id]
        used_npis = []

        # Get wins and sort by NPI
        win_npis = sorted(
            [(npi, won) for npi, won in team_data["all_game_npis"] if won],
            key=lambda x: x[0],
            reverse=True,
        )

        # Process wins
        for rank, (win_npi, won) in enumerate(win_npis, 1):
            if rank <= 15 or win_npi >= initial_npi:
                used_npis.append(win_npi)

        # Process losses
        loss_npis = sorted(
            [(npi, won) for npi, won in team_data["all_game_npis"] if not won],
            key=lambda x: x[0],
        )

# Include the worst loss and any losses with identical NPI
        included_loss_npis = set()
        if loss_npis:
            worst_loss_npi = loss_npis[0][0]
            included_loss_npis.add(worst_loss_npi)
            for loss_npi, _ in loss_npis:
                if loss_npi == worst_loss_npi:
                    used_npis.append(loss_npi)

        # Include all other losses below initial NPI, including identical NPIs
        processed_npi_values = set()
        for loss_npi, _ in loss_npis:
            if loss_npi < initial_npi and loss_npi not in included_loss_npis:
                if loss_npi not in processed_npi_values:
                    processed_npi_values.add(loss_npi)
                    for other_loss_npi, _ in loss_npis:
                        if other_loss_npi == loss_npi:
                            used_npis.append(other_loss_npi)

        if used_npis:
            team_data["game_npis"] = used_npis
            team_data["npi"] = sum(used_npis) / len(used_npis)
        else:
            team_data["game_npis"] = []
            team_data["npi"] = initial_npi

        # Set qualifying wins and losses
        team_data["qualifying_wins"] = len(
            [
                npi
                for npi in used_npis
                if npi
                in [win_npi for win_npi, won in team_data["all_game_npis"] if won]
            ]
        )
        team_data["qualifying_losses"] = len(
            [
                npi
                for npi in used_npis
                if npi
                in [loss_npi for loss_npi, won in team_data["all_game_npis"] if not won]
            ]
        )

    return teams


def load_games(base_path, year, valid_teams):
    games = []
    seen_games = set()
    year_str = str(year)
    games_path = base_path / year_str / "games.txt"
    zero_zero_count = 0
    duplicate_count = 0
    skipped_due_to_invalid_teams = 0

    try:
        with open(games_path, "r") as file:
            for line in file:
                try:
                    cols = line.strip().split(",")
                    if len(cols) < 8:
                        continue

                    # Extract game data
                    date = cols[0].strip()
                    team1_id = cols[2].strip()
                    team2_id = cols[5].strip()
                    team1_score = int(cols[4])
                    team2_score = int(cols[7])

                    # Skip 0-0 games
                    if team1_score == 0 and team2_score == 0:
                        zero_zero_count += 1
                        continue

                    # Skip games where either team is not in the valid_teams dictionary
                    if team1_id not in valid_teams or team2_id not in valid_teams:
                        skipped_due_to_invalid_teams += 1
                        continue

                    # Create a unique game identifier (ordered team IDs + date)
                    game_id = tuple(sorted([team1_id, team2_id]) + [date])

                    # Skip duplicates
                    if game_id in seen_games:
                        duplicate_count += 1
                        continue

                    seen_games.add(game_id)
                    games.append(
                        {
                            "date": date,
                            "team1_id": team1_id,
                            "team2_id": team2_id,
                            "team1_score": team1_score,
                            "team2_score": team2_score,
                        }
                    )

                except Exception as e:
                    continue

        print(f"Game Loading Statistics:")
        print(f"Total games loaded: {len(games)}")
        print(f"Skipped 0-0 games: {zero_zero_count}")
        print(f"Skipped duplicates: {duplicate_count}")
        print(f"Skipped due to invalid teams: {skipped_due_to_invalid_teams}")

    except Exception as e:
        print(f"Error loading games: {e}")

    return games


def save_npi_results_to_csv(teams):
    """Write results for an iteration to CSV."""
    data_path = Path(__file__).parent / "data" / "2025" / "npi.csv"
    
    # Load old rankings and find max rank
    old_rankings = {}
    max_rank = 0
    try:
        with open(data_path, "r", newline="") as csvfile:
            csv_data = list(csv.reader(csvfile))
            for row in csv_data[1:]:
                old_rankings[row[0]] = int(row[6])
            max_rank = max(max_rank, int(row[6]))
    except FileNotFoundError:
        pass

    active_teams = [team for team in teams.values() if team["has_games"]]
    active_teams = [dict(team, npi=float("{:.2f}".format(team["npi"]))) for team in teams.values() if team["has_games"]]
    sorted_teams = sorted(active_teams, key=lambda x: x["npi"], reverse=True)

    with open(data_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Team Name",
            "Games",
            "Wins",
            "Qualifying Wins",
            "Qualifying Losses",
            "NPI",
            "Rank",
            "Old Rank",
            "Rank Change",
        ])
        
        for rank, team in enumerate(sorted_teams, 1):
            if team["team_name"] in old_rankings:
                old_rank = old_rankings[team["team_name"]]
            else:
                old_rank = max_rank + 1
            rank_change = old_rank - rank
            rank_change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
            writer.writerow([
                team["team_name"],
                team["games"],
                team["wins"],
                team.get("qualifying_wins", 0),
                team.get("qualifying_losses", 0),
                "{:.2f}".format(float(team["npi"])),
                rank,
                old_rank,
                rank_change_str,
            ])


def process_games_bidirectional(file_path, teams_dict):
    """
    Process games from file and return bidirectional results list.
    Each game generates two rows - one from each team's perspective.
    Reformats date from yyyymmdd to mm/dd/yyyy.
    Only processes games where both teams exist in teams_dict.
    Uses column 4 and 6 values to determine Home/Road/Neutral status:
    - For team1: column 4 (1 = Home, 0 = Neutral)
    - For team2: column 6 (1 = Home, 0 = Neutral)
    """
    results = []
    skipped_games = 0
    processed_games = 0
    
    try:
        with open(file_path, "r") as file:
            for line in file:
                try:
                    cols = line.strip().split(",")
                    if len(cols) < 8:
                        continue
                        
                    team1_id = cols[2].strip()
                    team2_id = cols[5].strip()
                    
                    if team1_id not in teams_dict or team2_id not in teams_dict:
                        skipped_games += 1
                        continue
                        
                    date_str = cols[1].strip()
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    formatted_date = f"{month}/{day}/{year}"
                    
                    team1_score = int(cols[4])
                    team2_score = int(cols[7])
                    team1_location = int(cols[3].strip())
                    team2_location = int(cols[5].strip())
                    team1_name = teams_dict[team1_id]
                    team2_name = teams_dict[team2_id]
                    
                    if team1_score > team2_score:
                        team1_result = "W"
                        team2_result = "L"
                    elif team2_score > team1_score:
                        team1_result = "L"
                        team2_result = "W"
                    else:
                        team1_result = team2_result = "T"
                    
                    # Determine locations based on both indicators
                    team1_status = "Neutral" if team1_location == 0 else ("Home" if team1_location == 1 else "Road")
                    team2_status = "Neutral" if team2_location == 0 else ("Home" if team2_location == 1 else "Road")
                    
                    results.append(
                        f"{formatted_date},{team1_name},{team2_name},{team1_status},{team1_result}"
                    )
                    results.append(
                        f"{formatted_date},{team2_name},{team1_name},{team2_status},{team2_result}"
                    )
                    
                    processed_games += 1
                except Exception as e:
                    skipped_games += 1
                    continue
                    
        print(f"Processed {processed_games} games into {len(results)} bidirectional results")
        print(f"Skipped {skipped_games} games due to missing teams or invalid data")
    except Exception as e:
        print(f"Error processing games file: {e}")
        
    return results

def main():
    """Main entry point for the application."""
    data_path = Path(__file__).parent / "data"
    year = "2025"
    NUM_ITERATIONS = 99

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

    except Exception as e:
        print(f"Error processing: {e}")
        raise


if __name__ == "__main__":
    main()
