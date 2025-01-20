from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from pathlib import Path
from .game_simulation import simulate_game


@dataclass
class ConferenceTournament:
    conference: str
    total_teams: int
    byes: int
    reseeding: bool
    regular_season_end_date: str


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

    with open(conf_path, "r") as file:
        for line in file:
            team_id, team_name, conference = line.strip().split(",")
            teams[team_id] = ConferenceTeam(team_id, team_name, conference)

    return teams


def load_tournament_structures(
    base_path: Path, year: str
) -> Dict[str, ConferenceTournament]:
    """Load conference tournament structures."""
    tournaments = {}
    tourney_path = base_path / year / "conf_tournaments.txt"

    with open(tourney_path, "r") as file:
        next(file)  # Skip header
        for line in file:
            conf, teams, byes, reseeding, end_date = line.strip().split(",")
            tournaments[conf] = ConferenceTournament(
                conference=conf,
                total_teams=int(teams),
                byes=int(byes),
                reseeding=reseeding.upper() == "TRUE",
                regular_season_end_date=end_date
            )

    return tournaments


def calculate_conference_standings(
    season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]
) -> Dict[str, List[ConferenceStanding]]:
    """
    Calculate conference standings, using total record for C2C and conference games for others.
    """
    # Initialize records
    conference_records: Dict[str, Dict[str, Tuple[int, int]]] = {}
    c2c_records: Dict[str, Tuple[int, int]] = {}  # Special handling for C2C

    # Process each game
    for game in season_results:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]

        # Skip if either team isn't in our conference data
        if team1_id not in conference_teams or team2_id not in conference_teams:
            continue

        team1_conf = conference_teams[team1_id].conference
        team2_conf = conference_teams[team2_id].conference

        # Special handling for C2C teams - track all games
        for team_id, conf in [(team1_id, team1_conf), (team2_id, team2_conf)]:
            if conf == "C2C":
                if team_id not in c2c_records:
                    c2c_records[team_id] = [0, 0]  # [wins, losses]

        if team1_conf == "C2C" and team1_id in c2c_records:
            if game["team1_score"] > game["team2_score"]:
                c2c_records[team1_id][0] += 1
            else:
                c2c_records[team1_id][1] += 1

        if team2_conf == "C2C" and team2_id in c2c_records:
            if game["team2_score"] > game["team1_score"]:
                c2c_records[team2_id][0] += 1
            else:
                c2c_records[team2_id][1] += 1

        # Regular conference standings for all other conferences
        if team1_conf == team2_conf and team1_conf != "C2C":
            if team1_conf not in conference_records:
                conference_records[team1_conf] = {}

            # Initialize team records if needed
            for team_id in [team1_id, team2_id]:
                if team_id not in conference_records[team1_conf]:
                    conference_records[team1_conf][team_id] = [0, 0]  # [wins, losses]

            # Update conference records
            if game["team1_score"] > game["team2_score"]:
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
        conference_standings["C2C"] = c2c_standings

    return conference_standings


def simulate_tournament_game(
    team1_id: str,
    team2_id: str,
    team_data: Dict[str, Tuple[float, float]],
    game_date: str,
    team1_home: bool = False,
    team2_home: bool = False,
    conference: str = "",
    round_name: str = "",
) -> dict:
    """Simulate a single tournament game."""
    # Apply home court advantage
    home_advantage = 0.0
    if team1_home:
        home_advantage = -3.5  # Negative because sim func assumes team2 is home
    elif team2_home:
        home_advantage = 3.5
        
    result = simulate_game(team_data, team1_id, team2_id, home_advantage)

    return {
        "game_id": f"CONF_TOURNAMENT_{conference}_{round_name}",
        "date": game_date,
        "team1_id": team1_id,
        "team2_id": team2_id,
        "team1_home": 1 if team1_home else -1,
        "team2_home": 1 if team2_home else -1,
        "team1_score": (
            result.winning_score
            if result.winner_id == team1_id
            else result.losing_score
        ),
        "team2_score": (
            result.winning_score
            if result.winner_id == team2_id
            else result.losing_score
        ),
        "conference": conference,
        "round": round_name,
    }

def simulate_conference_tournaments(
    conference_teams: Dict[str, ConferenceTeam],
    tournament_structures: Dict[str, ConferenceTournament],
    conference_standings: Dict[str, List[ConferenceStanding]],
    team_data: Dict[str, Tuple[float, float]],
    completed_games: List[dict]
) -> Tuple[List[dict], Dict[str, str]]:
    """
    Simulate remaining conference tournament games accounting for already completed games.
    
    Args:
        conference_teams: Dictionary mapping team IDs to ConferenceTeam objects
        tournament_structures: Dictionary mapping conference IDs to tournament structures
        conference_standings: Dictionary mapping conference IDs to ordered standings
        team_data: Dictionary mapping team IDs to rating data
        completed_games: List of all completed games
        
    Returns:
        Tuple containing:
        - List of simulated tournament games
        - Dictionary mapping conference IDs to champion team IDs
    """
    all_tournament_games = []
    conference_champions = {}
    
    for conf, structure in tournament_structures.items():
        if structure.total_teams == 0:  # Skip conferences with no tournament
            continue

        if conf not in conference_standings:
            continue

        standings = conference_standings[conf]
        if len(standings) < structure.total_teams:
            continue

        # Get tournament teams
        tournament_teams = standings[: structure.total_teams]
        first_round_teams = tournament_teams[structure.byes :]
        bye_teams = tournament_teams[: structure.byes]

        # First round matchups
        remaining_teams = []
        if len(first_round_teams) > 0:
            tournament_date = str(int(structure.regular_season_end_date) + 1).zfill(8)
            
            # Create first round matchups (highest vs lowest seed)
            matchups = pair_teams_by_seed(first_round_teams)
            # Count total teams still in tournament (playing + byes)
            total_remaining = len(first_round_teams) + len(bye_teams)
            round_name = get_round_name(total_remaining)
            
            for higher_seed, lower_seed in matchups:
                game = simulate_tournament_game(
                    higher_seed.team_id,
                    lower_seed.team_id,
                    team_data,
                    tournament_date,
                    team1_home=True,  # Higher seed gets home court
                    team2_home=False,
                    conference=conf,
                    round_name=round_name,
                )
                all_tournament_games.append(game)

                # Add winner to remaining teams
                winner_id = (
                    higher_seed.team_id
                    if game["team1_score"] > game["team2_score"]
                    else lower_seed.team_id
                )
                remaining_teams.append(winner_id)

            # Increment date for next round
            tournament_date = str(int(tournament_date) + 1).zfill(8)

        # Add bye teams to remaining teams
        remaining_teams.extend([team.team_id for team in bye_teams])

        # Continue tournament until champion is crowned
        while len(remaining_teams) > 1:
            next_round_teams = []
            
            # Get current teams' standings objects for seeding
            current_team_standings = [
                next(s for s in standings if s.team_id == team_id)
                for team_id in remaining_teams
            ]
            
            if structure.reseeding:
                # Re-seed remaining teams based on original seed
                current_team_standings.sort(
                    key=lambda x: standings.index(x)
                )
                
            # Create matchups based on seeds
            matchups = pair_teams_by_seed(current_team_standings)
            round_name = get_round_name(len(remaining_teams))
            
            for higher_seed, lower_seed in matchups:
                is_championship = len(remaining_teams) == 2
                team1_home = not is_championship  # Neutral site for championship
                
                game = simulate_tournament_game(
                    higher_seed.team_id,
                    lower_seed.team_id,
                    team_data,
                    tournament_date,
                    team1_home=team1_home,
                    team2_home=False,
                    conference=conf,
                    round_name=round_name,
                )
                all_tournament_games.append(game)

                # Add winner to next round
                winner_id = (
                    higher_seed.team_id
                    if game["team1_score"] > game["team2_score"]
                    else lower_seed.team_id
                )
                next_round_teams.append(winner_id)

            remaining_teams = next_round_teams
            tournament_date = str(int(tournament_date) + 1).zfill(8)

        # Record conference champion
        if remaining_teams:
            conference_champions[conf] = remaining_teams[0]

    return all_tournament_games, conference_champions

def pair_teams_by_seed(teams: List[ConferenceStanding]) -> List[Tuple[ConferenceStanding, ConferenceStanding]]:
    """
    Create matchups pairing highest and lowest seeds.
    
    Args:
        teams: List of teams in seed order (best to worst)
        
    Returns:
        List of (higher_seed, lower_seed) tuples
    """
    num_teams = len(teams)
    matchups = []
    
    for i in range(num_teams // 2):
        matchups.append((teams[i], teams[-(i + 1)]))
        
    return matchups

def get_round_name(remaining_teams: int) -> str:
    """
    Get the name of the tournament round based on number of teams remaining.
    Args:
        remaining_teams: Total number of teams still in tournament (including byes)
        
    Returns:
        String indicating tournament round:
        - 2 teams = Final
        - 4 teams = Semifinal
        - 5-8 teams = Quarterfinal 
        - 9+ teams = Other
    """
    if remaining_teams == 2:
        return "Final"
    elif remaining_teams == 4:
        return "Semifinal"
    elif 5 <= remaining_teams <= 8:
        return "Quarterfinal"
    else:
        return "Other"

def get_completed_tournament_games(
    games: List[dict],
    regular_season_end_date: str,
    conference: str
) -> List[dict]:
    """
    Extract completed tournament games for a specific conference.
    
    Args:
        games: List of all completed games
        regular_season_end_date: Date string marking end of regular season (YYYYMMDD)
        conference: Conference identifier to filter games
        
    Returns:
        List of completed tournament games for the conference
    """
    return [
        game for game in games
        if (game["date"] > regular_season_end_date and 
            game.get("conference") == conference)
    ]

def determine_tournament_state(
    completed_games: List[dict],
    tournament_structure: ConferenceTournament,
    standings: List[ConferenceStanding]
) -> Tuple[List[str], List[str], str]:
    """
    Determine the current state of a conference tournament.
    
    Args:
        completed_games: List of completed tournament games for this conference
        tournament_structure: Tournament configuration
        standings: Conference standings in seed order
        
    Returns:
        Tuple containing:
        - List of team IDs still active in tournament
        - List of team IDs with first round byes
        - Date to use for next game (based on last completed game)
    """
    if not completed_games:
        # Tournament hasn't started - get initial tournament teams and byes
        tournament_teams = standings[:tournament_structure.total_teams]
        bye_teams = [team.team_id for team in tournament_teams[:tournament_structure.byes]]
        active_teams = [team.team_id for team in tournament_teams[tournament_structure.byes:]]
        return active_teams, bye_teams, "20250302"  # Start date
        
    # Sort games chronologically
    completed_games.sort(key=lambda x: x["date"])
    next_game_date = str(int(completed_games[-1]["date"]) + 1).zfill(8)
    
    # Track winners from each completed game
    winners = []
    for game in completed_games:
        winner_id = (
            game["team1_id"] 
            if game["team1_score"] > game["team2_score"]
            else game["team2_id"]
        )
        winners.append(winner_id)
    
    # Get teams that had byes and haven't played yet
    tournament_teams = standings[:tournament_structure.total_teams]
    bye_teams = [team.team_id for team in tournament_teams[:tournament_structure.byes]]
    all_played_teams = set()
    for game in completed_games:
        all_played_teams.add(game["team1_id"])
        all_played_teams.add(game["team2_id"])
    unplayed_bye_teams = [team_id for team_id in bye_teams if team_id not in all_played_teams]
    
    # Active teams are winners of last round plus unplayed bye teams
    round_sizes = get_tournament_round_sizes(len(tournament_teams), tournament_structure.byes)
    current_round_size = next(
        size for size in round_sizes 
        if size >= len(winners) + len(unplayed_bye_teams)
    )
    active_teams = winners[-(current_round_size - len(unplayed_bye_teams)):] + unplayed_bye_teams
    
    return active_teams, [], next_game_date

def get_tournament_round_sizes(total_teams: int, byes: int) -> List[int]:
    """
    Calculate number of teams in each round of tournament.
    
    Args:
        total_teams: Total teams in tournament
        byes: Number of first round byes
        
    Returns:
        List of integers representing teams per round, from first round to championship
    """
    first_round_teams = total_teams - byes
    if first_round_teams % 2 != 0:
        raise ValueError("First round teams must be even number")
        
    sizes = []
    remaining = first_round_teams // 2 + byes
    while remaining > 1:
        sizes.append(remaining)
        remaining = remaining // 2
    sizes.append(2)  # Championship game
    return sizes