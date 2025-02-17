import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from myapp.game_simulation import GameResult, simulate_game

from .conf_tournaments import (
    calculate_conference_standings,
    load_conference_data,
    load_tournament_structures,
    simulate_conference_tournaments,
)
from .elo_simulator import EloSimulator
from .ncaa_tournament_simulator import (
    NCAATournamentSimulator,
    TeamTournamentStats,
    save_tournament_stats,
)


def calculate_win_probability(
    team_data: Dict[str, Tuple[float, float]],
    team_a_id: str,
    team_b_id: str,
    home_advantage: float = 3.5,
) -> Dict[str, float]:
    """
    Calculate win probabilities for two teams based on their adjusted efficiency margins and tempo.

    Args:
        team_data: Dictionary mapping team_id to tuple of (adjEM, adjT)
        team_a_id: ID of first team
        team_b_id: ID of second team
        home_advantage: Points to adjust for home court advantage (default 3.5)

    Returns:
        Dictionary mapping team_ids to their win probabilities
    """
    if team_a_id not in team_data or team_b_id not in team_data:
        raise ValueError("One or both team IDs not found in data")

    team_a_stats = team_data[team_a_id]
    team_b_stats = team_data[team_b_id]

    # Calculate point differential
    adj_em_diff = team_a_stats[0] - team_b_stats[0]
    avg_tempo = (team_a_stats[1] + team_b_stats[1]) / 200

    # Calculate expected point differential
    point_diff = adj_em_diff * avg_tempo - home_advantage  # Assuming team B is home

    # Calculate win probability using normal distribution
    # Using 11 points as standard deviation per KenPom
    sigma = 11
    z_score = (0 - point_diff) / (sigma * math.sqrt(2))
    team_b_win_prob = 0.5 * (1 + math.erf(z_score))
    team_a_win_prob = 1 - team_b_win_prob

    return {team_a_id: team_a_win_prob, team_b_id: team_b_win_prob}


def load_efficiency_data(base_path: Path, year: int) -> Dict[str, Tuple[float, float]]:
    """
    Load team efficiency data from the specified file.

    Args:
        base_path: Base path to data directory
        year: Year to load data for

    Returns:
        Dictionary mapping team_id to tuple of (adjEM, adjT)
    """
    team_data = {}
    file_path = base_path / str(year) / "eff.txt"

    try:
        with open(file_path, "r") as file:
            for line in file:
                try:
                    team_id, adj_em, adj_t = line.strip().split(",")
                    team_data[team_id.strip()] = (float(adj_em), float(adj_t))
                except Exception as e:
                    continue
        print(f"Loaded efficiency data for {len(team_data)} teams")
        return team_data
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find efficiency data at {file_path}")


def predict_and_simulate_game(
    base_path: Path, team_a_id: str, team_b_id: str, year: int
) -> Tuple[Dict[str, float], GameResult]:
    """
    Predict win probabilities and simulate a game result.

    Args:
        base_path: Base path to data directory
        team_a_id: ID of first team
        team_b_id: ID of second team
        year: Year to use for predictions

    Returns:
        Tuple of (win probabilities dict, GameResult object)
    """
    team_data = load_efficiency_data(base_path, year)
    probabilities = calculate_win_probability(team_data, team_a_id, team_b_id)
    result = simulate_game(team_data, team_a_id, team_b_id, 0)
    return probabilities, result


def load_skip_list(base_path: Path, year: str) -> Set[Tuple]:
    """Load list of games to skip using the same format as games.txt"""
    skip_path = base_path / year / "skip_games.txt"
    skip_games = set()

    try:
        with open(skip_path, "r") as file:
            for line in file:
                try:
                    cols = line.strip().split(",")
                    if len(cols) < 8:
                        continue

                    game_id = cols[0].strip()
                    date = cols[1].strip()
                    team1_id = cols[2].strip()
                    team2_id = cols[5].strip()

                    # Create the same game key format as in load_all_games
                    game_key = tuple(sorted([team1_id, team2_id]) + [date])
                    skip_games.add(game_key)
                except Exception:
                    continue
    except FileNotFoundError:
        return set()

    return skip_games


def load_all_games(
    base_path: Path, year: str, valid_teams: Dict[str, str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load games from games.txt, separating into completed and future games.
    Returns tuple of (completed_games, future_games)
    """
    completed_games = []
    future_games = []
    seen_games = set()
    skip_games = load_skip_list(base_path, year)

    games_path = base_path / year / "games.txt"

    try:
        with open(games_path, "r") as file:
            for line in file:
                try:
                    cols = line.strip().split(",")
                    if len(cols) < 8:
                        continue

                    # Extract game data
                    game_id = cols[0].strip()
                    date = cols[1].strip()
                    team1_id = cols[2].strip()
                    home1 = int(cols[3].strip())
                    score1 = int(cols[4].strip())
                    team2_id = cols[5].strip()
                    home2 = int(cols[6].strip())
                    score2 = int(cols[7].strip())

                    # Skip games with invalid teams
                    if team1_id not in valid_teams or team2_id not in valid_teams:
                        continue

                    # Create unique game identifier
                    game_key = tuple(sorted([team1_id, team2_id]) + [date])
                    if game_key in seen_games or game_key in skip_games:
                        continue
                    seen_games.add(game_key)

                    game_data = {
                        "game_id": game_id,
                        "date": date,
                        "team1_id": team1_id,
                        "team2_id": team2_id,
                        "team1_home": home1,
                        "team2_home": home2,
                        "team1_score": score1,
                        "team2_score": score2,
                    }

                    # Sort into completed vs future games
                    if score1 == 0 and score2 == 0:
                        future_games.append(game_data)
                    else:
                        completed_games.append(game_data)

                except Exception as e:
                    continue

        print(
            f"Loaded {len(completed_games)} completed games and {len(future_games)} future games"
        )
        return completed_games, future_games

    except Exception as e:
        print(f"Error loading games: {e}")
        return [], []


def simulate_full_season(
    base_path: Path,
    year: str,
    valid_teams: Dict[str, str],
    team_data: Dict[str, Tuple[float, float]],
) -> bool:
    """
    Simulate remaining regular season games and conference tournaments.

    Args:
        base_path: Base path to data directory
        year: str: Year to simulate
        valid_teams: Dictionary mapping team IDs to names
        team_data: Dictionary mapping team_id to tuple of (adjEM, adjT)

    Returns:
        bool: True if simulation completed successfully
    """
    try:
        # Initialize Elo simulator
        simulator = EloSimulator()
        simulator.initialize_ratings(team_data)

        # Initialize lists to store results
        all_results = []

        # Load conference data
        conference_teams = load_conference_data(base_path, year)
        tournament_structures = load_tournament_structures(base_path, year)

        # Load and separate games
        completed_games, future_games = load_all_games(base_path, year, valid_teams)
        simulated_regular_season = []

        # Simulate remaining regular season games
        for game in future_games:
            try:
                team_a_id = game["team1_id"]
                team_b_id = game["team2_id"]
                team_b_is_home = game["team2_home"] == 1

                result = simulator.simulate_game(team_a_id, team_b_id, team_b_is_home)

                simulated_game = {
                    "game_id": game["game_id"],
                    "date": game["date"],
                    "team1_id": game["team1_id"],
                    "team2_id": game["team2_id"],
                    "team1_home": game["team1_home"],
                    "team2_home": game["team2_home"],
                    "team1_score": (
                        result.winning_score
                        if result.winner_id == team_a_id
                        else result.losing_score
                    ),
                    "team2_score": (
                        result.winning_score
                        if result.winner_id == team_b_id
                        else result.losing_score
                    ),
                }
                simulated_regular_season.append(simulated_game)

            except Exception as e:
                print(f"Error simulating regular season game {game['game_id']}: {e}")
                continue

        # Calculate conference standings using all regular season games
        all_regular_season_games = completed_games + simulated_regular_season
        conference_standings = calculate_conference_standings(
            all_regular_season_games, conference_teams, team_data
        )

        # Simulate conference tournaments
        tournament_games, _ = simulate_conference_tournaments(
            conference_teams=conference_teams,
            tournament_structures=tournament_structures,
            conference_standings=conference_standings,
            team_data=team_data,
            completed_games=all_regular_season_games,
        )

        # Combine all results
        all_results = completed_games + simulated_regular_season + tournament_games

        # Save all results
        output_path = base_path / year / "season_results.txt"
        with open(output_path, "w") as file:
            for game in all_results:
                line = f"{game['game_id']},{game['date']},{game['team1_id']:>6},{game['team1_home']:>3},{game['team1_score']:>4},{game['team2_id']:>6},{game['team2_home']:>3},{game['team2_score']:>4}\n"
                file.write(line)

        return True

    except Exception as e:
        print(f"Error in season simulation inside simulate_full_season(): {e}")
        return False


def simulate_multiple_tournaments(
    base_path: Path,
    year: str,
    num_sims: int,
    team_data: Dict[str, Tuple[float, float]],
    valid_teams: Dict[str, str],
) -> bool:
    """Run multiple tournament simulations and generate statistics"""
    try:
        games_file = base_path / year / "tourn_games.txt"
        stats_file = base_path / year / "tournament_stats.csv"

        # Initialize empty stats collection
        all_team_stats: Dict[str, List[TeamTournamentStats]] = defaultdict(list)

        # Run simulations
        for sim_num in range(num_sims):
            # Create fresh simulator for each run
            simulator = EloSimulator()
            simulator.initialize_ratings(team_data)
            tourney = NCAATournamentSimulator(simulator)

            # Load fresh tournament games
            tourney.load_tournament_games(games_file)

            # Simulate each round
            rounds = [64, 32, 16, 8, 4, 2]
            for round_num in rounds:
                tourney.simulate_round(round_num, team_data)
                if round_num > 2:
                    tourney.generate_next_round_games(round_num)

            # Record final stats for this simulation
            current_sim_stats = tourney.get_tournament_stats()
            for team_id, stats in current_sim_stats.items():
                all_team_stats[team_id].append(stats)

            # Print progress
            if (sim_num + 1) % 100 == 0:
                print(f"Completed {sim_num + 1} simulations")

        # Save aggregated stats
        save_tournament_stats(stats_file, all_team_stats, valid_teams, num_sims)
        return True

    except Exception as e:
        print(f"Error in tournament simulations: {e}")
        return False
