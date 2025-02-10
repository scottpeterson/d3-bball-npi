from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .game_simulation import simulate_game


@dataclass
class ConferenceTournament:
    conference: str
    total_teams: int
    byes: int
    reseeding: bool
    regular_season_end_date: str
    provisional_teams: Set[str]


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
    tournaments = {}
    path = base_path / year / "conf_tournaments.txt"

    with open(path, "r") as file:
        header = next(file).strip().split(",")
        has_provisional = "provisional_teams" in header

        for line in file:
            fields = line.strip().split(",")
            conf, teams, byes, reseeding, end_date = fields[:5]
            provisional_teams = (
                set(fields[5].split(";"))
                if has_provisional and len(fields) > 5 and fields[5]
                else set()
            )

            tournaments[conf] = ConferenceTournament(
                conference=conf,
                total_teams=int(teams),
                byes=int(byes),
                reseeding=reseeding.upper() == "TRUE",
                regular_season_end_date=end_date,
                provisional_teams=provisional_teams,
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
    completed_games: List[dict],
) -> Tuple[List[dict], Dict[str, str]]:
    all_tournament_games = []
    conference_champions = {}

    for conf, structure in tournament_structures.items():
        if structure.total_teams == 0 or conf not in conference_standings:
            continue

        standings = conference_standings[conf]
        if len(standings) < structure.total_teams:
            continue

        # Get current tournament state
        conf_completed_games = get_completed_tournament_games(
            completed_games, structure.regular_season_end_date, conf
        )
        remaining_teams, bye_teams, tournament_date = determine_tournament_state(
            conf_completed_games, structure, standings[: structure.total_teams]
        )

        # Handle first round if tournament hasn't started
        if not conf_completed_games and remaining_teams:
            # Create first round matchups (highest vs lowest seed)
            first_round_standings = [
                s
                for s in standings[: structure.total_teams]
                if s.team_id in remaining_teams
            ]
            matchups = pair_teams_by_seed(first_round_standings)
            round_name = get_round_name(len(remaining_teams) + len(bye_teams))

            next_round_teams = []
            for higher_seed, lower_seed in matchups:
                # Handle C2C conference venue logic
                team1_home = True  # Default: higher seed is home
                team2_home = False
                if conf == "C2C":
                    if lower_seed.team_id == "236":
                        # Mt. Mary is lower seed - swap home teams
                        team1_home = False
                        team2_home = True
                    elif higher_seed.team_id != "236":
                        # Neither team is Mt. Mary - neutral site
                        team1_home = False
                        team2_home = False
                game = simulate_tournament_game(
                    higher_seed.team_id,
                    lower_seed.team_id,
                    team_data,
                    tournament_date,
                    team1_home=team1_home,
                    team2_home=team2_home,
                    conference=conf,
                    round_name=round_name,
                )
                all_tournament_games.append(game)
                winner_id = (
                    higher_seed.team_id
                    if game["team1_score"] > game["team2_score"]
                    else lower_seed.team_id
                )
                next_round_teams.append(winner_id)

            remaining_teams = next_round_teams + [t for t in bye_teams]
            tournament_date = str(int(tournament_date) + 1).zfill(8)

        # Continue tournament until champion is crowned
        while len(remaining_teams) > 1:
            current_team_standings = [
                next(s for s in standings if s.team_id == team_id)
                for team_id in remaining_teams
            ]

            if structure.reseeding:
                current_team_standings.sort(key=lambda x: standings.index(x))

            matchups = pair_teams_by_seed(current_team_standings)
            round_name = get_round_name(len(remaining_teams))
            next_round_teams = []

            for higher_seed, lower_seed in matchups:
                # Handle C2C conference venue logic
                team1_home = True  # Default: higher seed is home
                team2_home = False
                if conf == "C2C":
                    if lower_seed.team_id == "236":
                        # Mt. Mary is lower seed - swap home teams
                        team1_home = False
                        team2_home = True
                    elif higher_seed.team_id != "236":
                        # Neither team is Mt. Mary - neutral site
                        team1_home = False
                        team2_home = False
                game = simulate_tournament_game(
                    higher_seed.team_id,
                    lower_seed.team_id,
                    team_data,
                    tournament_date,
                    team1_home=team1_home,
                    team2_home=team2_home,
                    conference=conf,
                    round_name=round_name,
                )
                all_tournament_games.append(game)
                winner_id = (
                    higher_seed.team_id
                    if game["team1_score"] > game["team2_score"]
                    else lower_seed.team_id
                )
                next_round_teams.append(winner_id)

            remaining_teams = next_round_teams
            tournament_date = str(int(tournament_date) + 1).zfill(8)

        champion_id = get_conference_champion(
            conf, structure.provisional_teams, all_tournament_games
        )
        if champion_id:
            conference_champions[conf] = champion_id

    return all_tournament_games, conference_champions


def pair_teams_by_seed(
    teams: List[ConferenceStanding],
) -> List[Tuple[ConferenceStanding, ConferenceStanding]]:
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
    games: List[dict], regular_season_end_date: str, conference: str
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
        game
        for game in games
        if (
            game["date"] > regular_season_end_date
            and game.get("conference") == conference
        )
    ]


def determine_tournament_state(
    completed_games: List[dict],
    tournament_structure: ConferenceTournament,
    standings: List[ConferenceStanding],
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
        tournament_teams = standings[: tournament_structure.total_teams]
        bye_teams = [
            team.team_id for team in tournament_teams[: tournament_structure.byes]
        ]
        active_teams = [
            team.team_id for team in tournament_teams[tournament_structure.byes :]
        ]
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
    tournament_teams = standings[: tournament_structure.total_teams]
    bye_teams = [team.team_id for team in tournament_teams[: tournament_structure.byes]]
    all_played_teams = set()
    for game in completed_games:
        all_played_teams.add(game["team1_id"])
        all_played_teams.add(game["team2_id"])
    unplayed_bye_teams = [
        team_id for team_id in bye_teams if team_id not in all_played_teams
    ]

    # Active teams are winners of last round plus unplayed bye teams
    round_sizes = get_tournament_round_sizes(
        len(tournament_teams), tournament_structure.byes
    )
    current_round_size = next(
        size for size in round_sizes if size >= len(winners) + len(unplayed_bye_teams)
    )
    active_teams = (
        winners[-(current_round_size - len(unplayed_bye_teams)) :] + unplayed_bye_teams
    )

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


def get_conference_champion(
    conference: str, provisional_teams: Set[str], tournament_games: List[dict]
) -> str:
    """Get eligible conference champion, accounting for provisional teams."""
    championship_game = next(
        game
        for game in reversed(tournament_games)
        if (game["conference"] == conference and game["round"] == "Final")
    )

    winner_id = (
        championship_game["team1_id"]
        if championship_game["team1_score"] > championship_game["team2_score"]
        else championship_game["team2_id"]
    )

    if winner_id not in provisional_teams:
        return winner_id

    # If winner is provisional, return runner-up if eligible
    runner_up = (
        championship_game["team2_id"]
        if championship_game["team1_id"] == winner_id
        else championship_game["team1_id"]
    )

    return runner_up if runner_up not in provisional_teams else None
