from collections import defaultdict
from .calculate_owp import calculate_owp
from .calculate_game_npi import calculate_game_npi


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
