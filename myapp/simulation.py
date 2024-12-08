from pathlib import Path
import math
from typing import Dict, List, Tuple, Any
import random
from dataclasses import dataclass

@dataclass
class GameResult:
    winner_id: str
    loser_id: str
    winning_score: int
    losing_score: int
    was_upset: bool

def simulate_game(team_data: Dict[str, Tuple[float, float]], team_a_id: str, team_b_id: str, 
                 home_advantage: float = 3.5) -> GameResult:
    """
    Simulate a game between two teams based on their win probabilities and efficiency metrics.
    
    Args:
        team_data: Dictionary mapping team_id to tuple of (adjEM, adjT)
        team_a_id: ID of first team
        team_b_id: ID of second team (home team)
        home_advantage: Points to adjust for home court advantage
        
    Returns:
        GameResult object containing winner, loser, scores, and whether it was an upset
    """
    # Get win probabilities first
    probs = calculate_win_probability(team_data, team_a_id, team_b_id, home_advantage)
    
    # Simulate winner based on probabilities
    teams = list(probs.keys())
    winner_id = random.choices(teams, weights=[probs[team_a_id], probs[team_b_id]], k=1)[0]
    loser_id = team_b_id if winner_id == team_a_id else team_a_id
    
    # Calculate expected point differential
    team_a_stats = team_data[team_a_id]
    team_b_stats = team_data[team_b_id]
    adj_em_diff = team_a_stats[0] - team_b_stats[0]
    avg_tempo = (team_a_stats[1] + team_b_stats[1]) / 2  # Average tempo of both teams
    expected_diff = adj_em_diff * (avg_tempo/100) - home_advantage
    
    # Generate random point differential using normal distribution
    actual_diff = random.gauss(expected_diff, 11)
    
    # Calculate expected points per possession for each team
    # Division by 100 because efficiency metrics are per 100 possessions
    base_efficiency = 102  # Average team scores about 102 points per 100 possessions
    team_a_efficiency = base_efficiency + (team_a_stats[0] / 2)  # Add half the efficiency margin
    team_b_efficiency = base_efficiency - (team_a_stats[0] / 2)  # Subtract half the efficiency margin
    
    # Calculate expected points based on tempo and efficiency
    possessions = avg_tempo  # This is already possessions per game
    team_a_expected = (team_a_efficiency * possessions) / 100
    team_b_expected = (team_b_efficiency * possessions) / 100
    expected_total = team_a_expected + team_b_expected
    
    # Add some random variation to total points
    total_points = random.gauss(expected_total, 10)
    
    if winner_id == team_a_id:
        actual_diff = abs(actual_diff)
    else:
        actual_diff = -abs(actual_diff)
    
    # Calculate individual team scores
    winner_score = round((total_points + abs(actual_diff)) / 2)
    loser_score = round((total_points - abs(actual_diff)) / 2)
    
    # Ensure scores are reasonable (minimum 55 points per team in modern college basketball)
    if winner_score < 55:
        winner_score = round(random.uniform(55, 65))
    if loser_score < 55:
        loser_score = round(random.uniform(55, winner_score - 3))
        
    # Cap extremely high scores (rare to see above 100 in college basketball)
    if winner_score > 100:
        winner_score = round(random.uniform(90, 100))
        loser_score = winner_score - round(abs(actual_diff))
    if loser_score > 95:
        loser_score = round(random.uniform(85, 95))
    
    # Determine if it was an upset
    was_upset = (winner_id == team_b_id and probs[team_a_id] > 0.5) or \
                (winner_id == team_a_id and probs[team_b_id] > 0.5)
    
    return GameResult(
        winner_id=winner_id,
        loser_id=loser_id,
        winning_score=winner_score,
        losing_score=loser_score,
        was_upset=was_upset
    )

def calculate_win_probability(team_data: Dict[str, Tuple[float, float]], team_a_id: str, team_b_id: str, home_advantage: float = 3.5) -> Dict[str, float]:
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
    
    return {
        team_a_id: team_a_win_prob,
        team_b_id: team_b_win_prob
    }

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

def predict_game(base_path: Path, team_a_id: str, team_b_id: str, year: int) -> Dict[str, float]:
    """
    Predict win probabilities for a game between two teams.
    
    Args:
        base_path: Base path to data directory
        team_a_id: ID of first team
        team_b_id: ID of second team
        year: Year to use for predictions
        
    Returns:
        Dictionary mapping team_ids to their win probabilities
    """
    team_data = load_efficiency_data(base_path, year)
    return calculate_win_probability(team_data, team_a_id, team_b_id)

def predict_and_simulate_game(base_path: Path, team_a_id: str, team_b_id: str, year: int) -> Tuple[Dict[str, float], GameResult]:
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
    result = simulate_game(team_data, team_a_id, team_b_id)
    return probabilities, result

def load_all_games(base_path: Path, year: str, valid_teams: Dict[str, str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Load games from games.txt, separating into completed and future games.
    Returns tuple of (completed_games, future_games)
    """
    completed_games = []
    future_games = []
    seen_games = set()
    
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
                    if game_key in seen_games:
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
                        "team2_score": score2
                    }
                    
                    # Sort into completed vs future games
                    if score1 == 0 and score2 == 0:
                        future_games.append(game_data)
                    else:
                        completed_games.append(game_data)
                        
                except Exception as e:
                    continue
                    
        print(f"Loaded {len(completed_games)} completed games and {len(future_games)} future games")
        return completed_games, future_games
        
    except Exception as e:
        print(f"Error loading games: {e}")
        return [], []

def simulate_season(base_path: Path, year: str, valid_teams: Dict[str, str], 
                   team_data: Dict[str, Tuple[float, float]], focus_team_id: str = "owu") -> bool:
    """
    Simulate remaining games in the season and save all results.
    Prints detailed results for specified team.
    """
    completed_games, future_games = load_all_games(base_path, year, valid_teams)
    
    # Track focus team's games
    focus_team_completed = []
    focus_team_simulated = []
    
    # Process completed games for focus team
    for game in completed_games:
        if game['team1_id'] == focus_team_id or game['team2_id'] == focus_team_id:
            focus_team_completed.append(game)
    
    # Simulate all future games
    simulated_results = []
    for game in future_games:
        try:
            team_a_id = game['team1_id']
            team_b_id = game['team2_id']
            home_advantage = 3.5 if game['team2_home'] == 1 else -3.5
            
            result = simulate_game(team_data, team_a_id, team_b_id, home_advantage)
            
            simulated_game = {
                "game_id": game['game_id'],
                "date": game['date'],
                "team1_id": game['team1_id'],
                "team2_id": game['team2_id'],
                "team1_home": game['team1_home'],
                "team2_home": game['team2_home'],
                "team1_score": result.winning_score if result.winner_id == team_a_id else result.losing_score,
                "team2_score": result.winning_score if result.winner_id == team_b_id else result.losing_score,
                "simulated": True
            }
            simulated_results.append(simulated_game)
            
            # Track focus team's simulated games
            if team_a_id == focus_team_id or team_b_id == focus_team_id:
                focus_team_simulated.append(simulated_game)
                
        except Exception as e:
            print(f"Error simulating game {game['game_id']}: {e}")
            continue
    
    # Print focus team's results
    print(f"\nResults for {valid_teams[focus_team_id]}:")
    print("\nCompleted Games:")
    print("-" * 60)
    for game in focus_team_completed:
        team1_name = valid_teams[game['team1_id']]
        team2_name = valid_teams[game['team2_id']]
        location = "home" if (game['team1_id'] == focus_team_id and game['team1_home'] == 1) or \
                           (game['team2_id'] == focus_team_id and game['team2_home'] == 1) else "away"
        print(f"{game['date']} - {location.upper()}")
        print(f"{team1_name}: {game['team1_score']}")
        print(f"{team2_name}: {game['team2_score']}\n")
    
    print("\nSimulated Games:")
    print("-" * 60)
    for game in focus_team_simulated:
        team1_name = valid_teams[game['team1_id']]
        team2_name = valid_teams[game['team2_id']]
        location = "home" if (game['team1_id'] == focus_team_id and game['team1_home'] == 1) or \
                           (game['team2_id'] == focus_team_id and game['team2_home'] == 1) else "away"
        print(f"{game['date']} - {location.upper()}")
        print(f"{team1_name}: {game['team1_score']}")
        print(f"{team2_name}: {game['team2_score']}\n")
    
    # Calculate season summary
    owu_wins = 0
    owu_losses = 0
    total_points_for = 0
    total_points_against = 0
    
    for game in focus_team_completed + focus_team_simulated:
        if game['team1_id'] == focus_team_id:
            owu_score = game['team1_score']
            opp_score = game['team2_score']
        else:
            owu_score = game['team2_score']
            opp_score = game['team1_score']
            
        if owu_score > opp_score:
            owu_wins += 1
        else:
            owu_losses += 1
            
        total_points_for += owu_score
        total_points_against += opp_score
    
    print("\nSeason Summary:")
    print("-" * 60)
    print(f"Record: {owu_wins}-{owu_losses}")
    if owu_wins + owu_losses > 0:
        print(f"Win Percentage: {owu_wins/(owu_wins + owu_losses):.3f}")
        print(f"Average Score: {total_points_for/(owu_wins + owu_losses):.1f}-{total_points_against/(owu_wins + owu_losses):.1f}")
    
    # Save all results
    all_results = completed_games + simulated_results
    output_path = base_path / year / "season_results.txt"
    try:
        with open(output_path, "w") as file:
            for game in all_results:
                line = f"{game['game_id']},{game['date']},{game['team1_id']:>6},{game['team1_home']:>3},{game['team1_score']:>4},{game['team2_id']:>6},{game['team2_home']:>3},{game['team2_score']:>4}\n"
                file.write(line)
        print(f"\nSaved {len(all_results)} total games to season_results.txt")
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return False