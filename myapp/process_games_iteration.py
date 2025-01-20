from collections import defaultdict
from .calculate_owp import calculate_owp
from .calculate_game_npi import calculate_game_npi


def process_games_iteration(games, valid_teams, previous_iteration_npis=None, iteration_number=1):
    owp = calculate_owp(games, valid_teams)
    
    # Set up opponent_npis early
    if iteration_number == 1:
        opponent_npis = {team_id: 50 for team_id in valid_teams}
    else:
        opponent_npis = {}
        for team_id in valid_teams:
            opponent_npis[team_id] = previous_iteration_npis.get(team_id, owp[team_id])

    # Initialize teams dict with all required keys
    teams = {
        team_id: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "npi": opponent_npis[team_id],
            "game_npis": [],
            "all_game_npis": [],
            "team_id": team_id,
            "team_name": team_name,
            "qualifying_wins": 0,
            "qualifying_losses": 0,
            "has_games": False,
        }
        for team_id, team_name in valid_teams.items()
    }

    # Combine first two passes - record stats and calculate game NPIs
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

        # Update basic stats
        teams[team1_id]["has_games"] = True
        teams[team2_id]["has_games"] = True
        teams[team1_id]["games"] += 1
        teams[team2_id]["games"] += 1

        # Update wins/losses and calculate NPIs in same pass
        if team1_score > team2_score:
            teams[team1_id]["wins"] += 1
            teams[team2_id]["losses"] += 1
            team1_won, team2_won = True, False
        elif team2_score > team1_score:
            teams[team2_id]["wins"] += 1
            teams[team1_id]["losses"] += 1
            team1_won, team2_won = False, True
        else:
            teams[team1_id]["ties"] += 1
            teams[team2_id]["ties"] += 1
            team1_won, team2_won = False, False

        # Calculate and store game NPIs
        team1_game_npi = calculate_game_npi(team1_won, opponent_npis[team2_id])
        team2_game_npi = calculate_game_npi(team2_won, opponent_npis[team1_id])
        teams[team1_id]["all_game_npis"].append((team1_game_npi, team1_won))
        teams[team2_id]["all_game_npis"].append((team2_game_npi, team2_won))

    # Optimized third pass: filter and calculate final NPIs
    for team_id, team_data in teams.items():
        if not team_data["has_games"]:
            continue

        initial_npi = opponent_npis[team_id]
        all_games = team_data["all_game_npis"]
        used_npis = []

        # Split and sort wins/losses once
        wins = []
        losses = []
        for npi, won in all_games:
            if won:
                wins.append(npi)
            else:
                losses.append(npi)
                
        # Sort once
        wins.sort(reverse=True)
        losses.sort()

        # Process top 15 wins and wins above initial NPI
        for i, win_npi in enumerate(wins):
            if i < 15 or win_npi >= initial_npi:
                used_npis.append(win_npi)

        # Process losses more efficiently
        if losses:
            # Add all instances of the worst loss
            worst_loss = losses[0]
            used_npis.extend(npi for npi in losses if npi == worst_loss)

            # Add all losses below initial NPI, grouping identical NPIs
            seen_npis = {worst_loss}
            for loss_npi in losses:
                if loss_npi < initial_npi and loss_npi not in seen_npis:
                    seen_npis.add(loss_npi)
                    used_npis.extend(npi for npi in losses if npi == loss_npi)

        # Calculate final NPI and stats
        if used_npis:
            team_data["game_npis"] = used_npis
            team_data["npi"] = sum(used_npis) / len(used_npis)
            # Count qualifying games more efficiently
            team_data["qualifying_wins"] = sum(1 for npi in used_npis if npi in wins)
            team_data["qualifying_losses"] = sum(1 for npi in used_npis if npi in losses)
        else:
            team_data["game_npis"] = []
            team_data["npi"] = initial_npi
            team_data["qualifying_wins"] = 0
            team_data["qualifying_losses"] = 0

    return teams
