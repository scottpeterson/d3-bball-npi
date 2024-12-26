from collections import defaultdict


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
