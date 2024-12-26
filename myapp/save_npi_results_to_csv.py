import csv
from pathlib import Path
from .process_games_bidirectional import process_games_bidirectional


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
    active_teams = [
        dict(team, npi=float("{:.2f}".format(team["npi"])))
        for team in teams.values()
        if team["has_games"]
    ]
    sorted_teams = sorted(active_teams, key=lambda x: x["npi"], reverse=True)

    with open(data_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Team Name",
                "Games",
                "Wins",
                "Qualifying Wins",
                "Qualifying Losses",
                "NPI",
                "Rank",
                "Old Rank",
                "Rank Change",
            ]
        )

        for rank, team in enumerate(sorted_teams, 1):
            if team["team_name"] in old_rankings:
                old_rank = old_rankings[team["team_name"]]
            else:
                old_rank = max_rank + 1
            rank_change = old_rank - rank
            rank_change_str = f"+{rank_change}" if rank_change > 0 else str(rank_change)
            writer.writerow(
                [
                    team["team_name"],
                    team["games"],
                    team["wins"],
                    team.get("qualifying_wins", 0),
                    team.get("qualifying_losses", 0),
                    "{:.2f}".format(float(team["npi"])),
                    rank,
                    old_rank,
                    rank_change_str,
                ]
            )
