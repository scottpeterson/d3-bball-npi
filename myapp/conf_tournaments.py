from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import datetime
from .game_simulation import simulate_game, GameResult

@dataclass
class ConferenceTournament:
    conference: str
    total_teams: int
    byes: int
    reseeding: bool

@dataclass
class ConferenceTeam:
    team_id: str
    team_name: str
    conference: str

@dataclass
class ConferenceStanding:
    team_id: str
    wins: int
    losses: int
    win_pct: float

def load_conference_data(base_path: Path, year: str) -> Dict[str, ConferenceTeam]:
    """Load team conference affiliations."""
    teams = {}
    conf_path = base_path / year / "conferences.txt"
    
    with open(conf_path, 'r') as file:
        for line in file:
            team_id, team_name, conference = line.strip().split(',')
            teams[team_id] = ConferenceTeam(team_id, team_name, conference)
    
    return teams

def load_tournament_structures(base_path: Path, year: str) -> Dict[str, ConferenceTournament]:
    """Load conference tournament structures."""
    tournaments = {}
    tourney_path = base_path / year / "conf_tournaments.txt"
    
    with open(tourney_path, 'r') as file:
        next(file)  # Skip header
        for line in file:
            conf, teams, byes, reseeding = line.strip().split(',')
            tournaments[conf] = ConferenceTournament(
                conference=conf,
                total_teams=int(teams),
                byes=int(byes),
                reseeding=reseeding.upper() == 'TRUE'
            )
    
    return tournaments

def calculate_conference_standings(season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]) -> Dict[str, List[ConferenceStanding]]:
    """
    Calculate conference standings, using total record for C2C and conference games for others.
    """
    # Initialize records
    conference_records: Dict[str, Dict[str, Tuple[int, int]]] = {}
    c2c_records: Dict[str, Tuple[int, int]] = {}  # Special handling for C2C
    
    # Process each game
    for game in season_results:
        team1_id = game['team1_id']
        team2_id = game['team2_id']
        
        # Skip if either team isn't in our conference data
        if team1_id not in conference_teams or team2_id not in conference_teams:
            continue
            
        team1_conf = conference_teams[team1_id].conference
        team2_conf = conference_teams[team2_id].conference
        
        # Special handling for C2C teams - track all games
        for team_id, conf in [(team1_id, team1_conf), (team2_id, team2_conf)]:
            if conf == 'C2C':
                if team_id not in c2c_records:
                    c2c_records[team_id] = [0, 0]  # [wins, losses]
        
        if team1_conf == 'C2C' and team1_id in c2c_records:
            if game['team1_score'] > game['team2_score']:
                c2c_records[team1_id][0] += 1
            else:
                c2c_records[team1_id][1] += 1
                
        if team2_conf == 'C2C' and team2_id in c2c_records:
            if game['team2_score'] > game['team1_score']:
                c2c_records[team2_id][0] += 1
            else:
                c2c_records[team2_id][1] += 1
        
        # Regular conference standings for all other conferences
        if team1_conf == team2_conf and team1_conf != 'C2C':
            if team1_conf not in conference_records:
                conference_records[team1_conf] = {}
            
            # Initialize team records if needed
            for team_id in [team1_id, team2_id]:
                if team_id not in conference_records[team1_conf]:
                    conference_records[team1_conf][team_id] = [0, 0]  # [wins, losses]
            
            # Update conference records
            if game['team1_score'] > game['team2_score']:
                conference_records[team1_conf][team1_id][0] += 1
                conference_records[team1_conf][team2_id][1] += 1
            else:
                conference_records[team1_conf][team2_id][0] += 1
                conference_records[team1_conf][team1_id][1] += 1
    
    # Convert records to standings
    conference_standings = {}
    
    # Process regular conference standings
    for conf in conference_records:
        standings = []
        for team_id, (wins, losses) in conference_records[conf].items():
            win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0
            standings.append(ConferenceStanding(team_id, wins, losses, win_pct))
        
        standings.sort(key=lambda x: (-x.win_pct, -x.wins))
        conference_standings[conf] = standings
    
    # Process C2C standings using total record
    if c2c_records:
        c2c_standings = []
        for team_id, (wins, losses) in c2c_records.items():
            win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0
            c2c_standings.append(ConferenceStanding(team_id, wins, losses, win_pct))
        
        c2c_standings.sort(key=lambda x: (-x.win_pct, -x.wins))
        conference_standings['C2C'] = c2c_standings
    
    return conference_standings

def simulate_tournament_game(team1_id: str, team2_id: str, team_data: Dict[str, Tuple[float, float]], 
                           game_date: str, game_counter: int) -> dict:
    """Simulate a single tournament game."""
    result = simulate_game(team_data, team1_id, team2_id)
    
    return {
        "game_id": f"CONF_TOURNAMENT_{game_counter}",
        "date": game_date,
        "team1_id": team1_id,
        "team2_id": team2_id,
        "team1_home": 0,  # Neutral site
        "team2_home": 0,  # Neutral site
        "team1_score": result.winning_score if result.winner_id == team1_id else result.losing_score,
        "team2_score": result.winning_score if result.winner_id == team2_id else result.losing_score
    }

from collections import defaultdict

def simulate_conference_tournaments(
    conference_teams: Dict[str, ConferenceTeam],
    tournament_structures: Dict[str, ConferenceTournament],
    conference_standings: Dict[str, List[ConferenceStanding]],
    team_data: Dict[str, Tuple[float, float]],
    tournament_start_date: str
) -> Tuple[List[dict], Dict[str, str]]:
    """
    Simulate all conference tournaments.
    Returns list of game results and dictionary of conference champions.
    """
    all_tournament_games = []
    conference_champions = {}
    game_counter = 1

    for conf, structure in tournament_structures.items():
        if structure.total_teams == 0:  # Skip conferences with no tournament
            continue

        if conf not in conference_standings:
            continue

        standings = conference_standings[conf]
        if len(standings) < structure.total_teams:
            continue

        # Get tournament teams
        tournament_teams = standings[:structure.total_teams]
        first_round_teams = tournament_teams[structure.byes:]
        bye_teams = tournament_teams[:structure.byes]

        # First round matchups
        remaining_teams = []
        if len(first_round_teams) > 0:
            tournament_date = tournament_start_date
            for i in range(0, len(first_round_teams), 2):
                if i + 1 < len(first_round_teams):
                    game = simulate_tournament_game(
                        first_round_teams[i].team_id,
                        first_round_teams[i+1].team_id,
                        team_data,
                        tournament_date,
                        game_counter
                    )
                    game_counter += 1
                    all_tournament_games.append(game)

                    # Add winner to remaining teams
                    winner_id = first_round_teams[i].team_id if game['team1_score'] > game['team2_score'] else first_round_teams[i+1].team_id
                    remaining_teams.append(winner_id)

                # Increment date for next round
                tournament_date = str(int(tournament_date) + 1).zfill(8)

        # Add bye teams to remaining teams
        remaining_teams.extend([team.team_id for team in bye_teams])

        # Continue tournament until champion is crowned
        while len(remaining_teams) > 1:
            next_round_teams = []
            if structure.reseeding:
                # Re-seed remaining teams based on original seed
                remaining_teams.sort(key=lambda x: standings.index(next(s for s in standings if s.team_id == x)))

            tournament_date = str(int(tournament_date) + 1).zfill(8)
            for i in range(0, len(remaining_teams), 2):
                if i + 1 < len(remaining_teams):
                    game = simulate_tournament_game(
                        remaining_teams[i],
                        remaining_teams[i+1],
                        team_data,
                        tournament_date,
                        game_counter
                    )
                    game_counter += 1
                    all_tournament_games.append(game)

                    # Add winner to next round
                    winner_id = remaining_teams[i] if game['team1_score'] > game['team2_score'] else remaining_teams[i+1]
                    next_round_teams.append(winner_id)

            remaining_teams = next_round_teams

        # Record conference champion
        if remaining_teams:
            conference_champions[conf] = remaining_teams[0]

    return all_tournament_games, conference_champions