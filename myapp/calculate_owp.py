from collections import defaultdict


def calculate_owp(games, valid_teams):
    # Pre-initialize records with known structure
    records = {team_id: {"wins": 0, "losses": 0, "games": 0} for team_id in valid_teams}

    # Build opponent records per team
    opponent_stats = {
        team_id: {"total_wins": 0, "total_losses": 0, "games": []}
        for team_id in valid_teams
    }

    # Single pass through games to build both records and collect opponent games
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

        # Update records
        records[team1_id]["games"] += 1
        records[team2_id]["games"] += 1

        if team1_score > team2_score:
            records[team1_id]["wins"] += 1
            records[team2_id]["losses"] += 1
        elif team2_score > team1_score:
            records[team2_id]["wins"] += 1
            records[team1_id]["losses"] += 1

        # Store game references for each team's opponents
        opponent_stats[team1_id]["games"].append((team2_id, game))
        opponent_stats[team2_id]["games"].append((team1_id, game))

    # Calculate OWP for each team
    owp = {}
    for team_id, opp_data in opponent_stats.items():
        opponents_total_wins = 0
        opponents_total_losses = 0

        # Process each opponent's record once
        for opp_id, game in opp_data["games"]:
            opp_record = records[opp_id]
            opp_wins = opp_record["wins"]
            opp_losses = opp_record["losses"]

            # Adjust for head-to-head results
            if game["team1_id"] == team_id:
                if game["team1_score"] > game["team2_score"]:
                    opp_losses -= 1
                elif game["team2_score"] > game["team1_score"]:
                    opp_wins -= 1
            else:
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
