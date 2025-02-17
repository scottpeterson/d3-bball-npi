import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .elo_simulator import EloSimulator


@dataclass
class TournamentGame:
    """Represents a tournament game with teams and scores"""

    massey_date_id: str
    date_string: str
    team1_id: str
    team1_loc: int
    team1_score: int
    team2_id: str
    team2_loc: int
    team2_score: int
    round: int
    winner_id: Optional[str] = None


@dataclass
class TeamTournamentStats:
    """Stats for a team in a single tournament run"""

    team_id: str
    games_won: int = 0
    exit_round: Optional[int] = None


class NCAATournamentSimulator:
    def __init__(self, simulator: "EloSimulator"):
        self.simulator = simulator
        self.games: List[TournamentGame] = []
        self.completed_games: List[TournamentGame] = []

    def load_tournament_games(self, filepath: Path) -> None:
        """Load existing tournament games from file"""
        self.games = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                game = TournamentGame(
                    massey_date_id=row[0],
                    date_string=row[1],
                    team1_id=row[2],
                    team1_loc=int(row[3]),
                    team1_score=int(row[4]),
                    team2_id=row[5],
                    team2_loc=int(row[6]),
                    team2_score=int(row[7]),
                    round=int(row[8]),
                )
                self.games.append(game)

    def simulate_round(
        self, current_round: int, team_data: Dict[str, Tuple[float, float]]
    ) -> List[TournamentGame]:
        """Simulate all games for a specific round"""
        round_games = [game for game in self.games if game.round == current_round]
        simulated_games = []

        for game in round_games:
            # Simulate the game
            result = self.simulator.simulate_game(
                game.team1_id, game.team2_id, team_b_home=(game.team2_loc == 1)
            )

            # Record the result
            if result.winner_id == game.team1_id:
                game.team1_score = result.winning_score
                game.team2_score = result.losing_score
            else:
                game.team1_score = result.losing_score
                game.team2_score = result.winning_score

            game.winner_id = result.winner_id
            simulated_games.append(game)
            self.completed_games.append(game)

        return simulated_games

    def generate_next_round_games(self, current_round: int) -> List[TournamentGame]:
        """Generate games for the next round based on winners"""
        next_round = current_round // 2
        # Only look at the games from the current round
        current_games = [g for g in self.completed_games if g.round == current_round]
        next_round_games = []

        # Calculate the date for the next round (add 4 days)
        base_date = datetime.strptime(current_games[0].date_string, "%Y%m%d")
        next_date = base_date + timedelta(days=4)
        next_date_str = next_date.strftime("%Y%m%d")

        # Generate new Massey ID (increment by 1)
        next_massey_id = str(int(current_games[0].massey_date_id) + 1)

        # Create matchups for next round
        for i in range(0, len(current_games), 2):
            if i + 1 < len(current_games):
                game1 = current_games[i]
                game2 = current_games[i + 1]

                # Create new game with winners
                new_game = TournamentGame(
                    massey_date_id=next_massey_id,
                    date_string=next_date_str,
                    team1_id=game1.winner_id,
                    team1_loc=-1,  # Neutral site
                    team1_score=0,
                    team2_id=game2.winner_id,
                    team2_loc=-1,  # Neutral site
                    team2_score=0,
                    round=next_round,
                )
                next_round_games.append(new_game)

        # Replace old games of this round with new ones
        self.games = [g for g in self.games if g.round != next_round] + next_round_games
        return next_round_games

    def save_results(self, filepath: Path) -> None:
        """Save all tournament results back to file"""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "MasseyDateId",
                    "DateString",
                    "TeamId",
                    "Loc",
                    "Score",
                    "TeamId",
                    "Loc",
                    "Score",
                    "ROUND",
                ]
            )

            for game in self.completed_games:
                writer.writerow(
                    [
                        game.massey_date_id,
                        game.date_string,
                        game.team1_id,
                        game.team1_loc,
                        game.team1_score,
                        game.team2_id,
                        game.team2_loc,
                        game.team2_score,
                        game.round,
                    ]
                )

    def get_tournament_stats(self) -> Dict[str, TeamTournamentStats]:
        """Get stats for the current tournament simulation"""
        current_sim_stats: Dict[str, TeamTournamentStats] = {}

        # Create stats objects for all teams and set their initial exit round to 64
        all_team_ids = set()
        for game in self.completed_games:
            all_team_ids.add(game.team1_id)
            all_team_ids.add(game.team2_id)

        for team_id in all_team_ids:
            current_sim_stats[team_id] = TeamTournamentStats(
                team_id=team_id,
                games_won=0,
                exit_round=64,  # Everyone starts in round 64
            )

        # Process games in order to properly count wins and exits
        for round_num in [64, 32, 16, 8, 4, 2]:
            round_games = [g for g in self.completed_games if g.round == round_num]

            for game in round_games:
                if game.winner_id:
                    # Record win
                    current_sim_stats[game.winner_id].games_won += 1

                    # Update exit rounds
                    loser_id = (
                        game.team2_id
                        if game.winner_id == game.team1_id
                        else game.team1_id
                    )
                    next_round = game.round // 2
                    current_sim_stats[game.winner_id].exit_round = next_round
                    current_sim_stats[loser_id].exit_round = game.round

        return current_sim_stats


def save_tournament_stats(
    filepath: Path,
    all_team_stats: Dict[str, List[TeamTournamentStats]],
    valid_teams: Dict[str, str],
    num_sims: int,
) -> None:
    """Save aggregated tournament statistics"""

    aggregate_stats = []
    for team_id, sim_stats in all_team_stats.items():
        # Calculate wins
        total_wins = sum(stat.games_won for stat in sim_stats)
        avg_wins = total_wins / num_sims

        # Count appearances in each round
        round_appearances = defaultdict(int)
        for stat in sim_stats:
            exit_round = stat.exit_round
            for round_num in [64, 32, 16, 8, 4, 2]:
                if round_num >= exit_round:
                    round_appearances[round_num] += 1

        # Calculate percentages
        round_percentages = {
            round_num: (count / num_sims) * 100
            for round_num, count in round_appearances.items()
        }

        aggregate_stats.append(
            {
                "team_id": team_id,
                "team_name": valid_teams[team_id],
                "avg_wins": round(avg_wins, 1),
                "total_wins": total_wins,
                "championships": round_appearances.get(
                    2, 0
                ),  # Champions exit in round 2
                "championship_pct": round(round_percentages.get(2, 0), 1),
                "final_four": round_appearances.get(4, 0),
                "final_four_pct": round(round_percentages.get(4, 0), 1),
                "elite_eight": round_appearances.get(8, 0),
                "elite_eight_pct": round(round_percentages.get(8, 0), 1),
                "sweet_sixteen": round_appearances.get(16, 0),
                "sweet_sixteen_pct": round(round_percentages.get(16, 0), 1),
                "round_32": round_appearances.get(32, 0),
                "round_32_pct": round(round_percentages.get(32, 0), 1),
                "round_64": round_appearances.get(64, 0),
                "round_64_pct": round(round_percentages.get(64, 0), 1),
            }
        )

    # Sort by championship percentage, then average wins
    aggregate_stats.sort(key=lambda x: (-x["championship_pct"], -x["avg_wins"]))

    # Save to CSV
    with open(filepath, "w", newline="") as f:
        fieldnames = [
            "team_name",
            "team_id",
            "avg_wins",
            "total_wins",
            "championships",
            "championship_pct",
            "final_four",
            "final_four_pct",
            "elite_eight",
            "elite_eight_pct",
            "sweet_sixteen",
            "sweet_sixteen_pct",
            "round_32",
            "round_32_pct",
            "round_64",
            "round_64_pct",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_stats)
