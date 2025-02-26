import functools
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

    # Build a dictionary with conference names as keys and their regular season end dates as values
    conference_end_dates = {
        conf: tournament.regular_season_end_date
        for conf, tournament in tournaments.items()
    }

    return tournaments, conference_end_dates


def calculate_conference_standings(
    season_results: List[dict],
    conference_teams: Dict[str, ConferenceTeam],
    team_data: Dict[str, Tuple[float, float]],
) -> Dict[str, List[ConferenceStanding]]:
    """
    Calculate conference standings:
    - C2C conference: Seed by NPI only
    - All other conferences:
        1. Conference win%
        2. Head-to-head record
        3. NPI
    """
    # Initialize records for non-C2C conferences
    conference_records: Dict[str, Dict[str, Tuple[int, int]]] = {}

    # Handle regular conference games
    for game in season_results:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]

        # Skip if either team isn't in our conference data
        if team1_id not in conference_teams or team2_id not in conference_teams:
            continue

        team1_conf = conference_teams[team1_id].conference
        team2_conf = conference_teams[team2_id].conference

        # Process conference games for non-C2C conferences
        if team1_conf == team2_conf and team1_conf != "C2C":
            if team1_conf not in conference_records:
                conference_records[team1_conf] = {}

            # Initialize team records if needed
            for team_id in [team1_id, team2_id]:
                if team_id not in conference_records[team1_conf]:
                    conference_records[team1_conf][team_id] = [0, 0]

            # Update conference records
            if game["team1_score"] > game["team2_score"]:
                conference_records[team1_conf][team1_id][0] += 1
                conference_records[team1_conf][team2_id][1] += 1
            else:
                conference_records[team1_conf][team2_id][0] += 1
                conference_records[team1_conf][team1_id][1] += 1

    # Convert records to standings
    conference_standings = {}

    # Process regular conference standings with tiebreakers
    for conf in conference_records:
        standings = []

        # Create initial standings
        for team_id, (wins, losses) in conference_records[conf].items():
            win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0
            npi = team_data[team_id][0] if team_id in team_data else 0.0
            standing = ConferenceStanding(team_id, wins, losses, win_pct)
            standing.npi = npi
            standings.append(standing)

        # Custom sort with tiebreakers
        def conference_sort_key(a, b):
            if abs(a.win_pct - b.win_pct) < 0.0001:  # Win% tie
                h2h_wins, h2h_losses = calculate_head_to_head(
                    season_results, a.team_id, b.team_id
                )
                if h2h_wins != h2h_losses:  # Head-to-head decides
                    return -1 if h2h_wins > h2h_losses else 1
                # If head-to-head tied, use NPI
                return -1 if a.npi > b.npi else 1
            return -1 if a.win_pct > b.win_pct else 1

        # Sort using custom comparison
        standings.sort(key=functools.cmp_to_key(conference_sort_key))
        conference_standings[conf] = standings

    # Process C2C standings using NPI only
    c2c_standings = []
    for team_id, team in conference_teams.items():
        if team.conference == "C2C":
            npi = team_data[team_id][0] if team_id in team_data else 0.0
            standing = ConferenceStanding(
                team_id, 0, 0, 0.0
            )  # wins/losses/win_pct not used
            standing.npi = npi
            c2c_standings.append(standing)

    # Sort C2C teams by NPI only
    c2c_standings.sort(key=lambda x: -x.npi)
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
    """
    Simulate conference tournaments accounting for games already played.

    Args:
        conference_teams: Dictionary mapping team IDs to conference info
        tournament_structures: Dictionary of conference tournament structures
        conference_standings: Dictionary of conference standings
        team_data: Team performance data (NPI, etc.)
        completed_games: List of all completed games

    Returns:
        Tuple of (tournament_games, conference_champions)
    """
    all_tournament_games = []
    conference_champions = {}

    # Load the base data (conference metadata and conference end dates)
    base_path = Path(__file__).parent / "data"
    tournament_structures, conference_end_dates = load_tournament_structures(
        base_path, "2025"
    )

    # Get ALL completed tournament games upfront
    all_completed_tournament_games = get_completed_tournament_games(
        games=completed_games,
        team_conference_data=conference_teams,
        conference_tournament_info=conference_end_dates,
    )

    for conf, structure in tournament_structures.items():
        if structure.total_teams == 0 or conf not in conference_standings:
            continue

        standings = conference_standings[conf]
        if len(standings) < structure.total_teams:
            continue

        # Filter tournament games for just this conference
        conf_completed_games = []
        for game in all_completed_tournament_games:
            team1_id = game.get("team1_id")
            team2_id = game.get("team2_id")

            team1 = conference_teams.get(team1_id)
            team2 = conference_teams.get(team2_id)

            # Skip if team info missing
            if not team1 or not team2:
                continue

            # Check if both teams are from current conference
            if team1.conference == conf and team2.conference == conf:
                # Add the conference and potentially the round information if missing
                if "conference" not in game:
                    game["conference"] = conf

                conf_completed_games.append(game)

        # Add already completed tournament games to our results
        all_tournament_games.extend(conf_completed_games)

        # Determine tournament state based on completed games and remaining teams
        remaining_teams, bye_teams, tournament_date = determine_tournament_state(
            conf_completed_games, structure, standings[: structure.total_teams]
        )

        # Handle first round if tournament hasn't started
        if not conf_completed_games and remaining_teams:
            first_round_standings = [
                s
                for s in standings[: structure.total_teams]
                if s.team_id in remaining_teams
            ]
            matchups = pair_teams_by_seed(first_round_standings)
            round_name = get_round_name(len(remaining_teams) + len(bye_teams))

            next_round_teams = []
            for higher_seed, lower_seed in matchups:
                team1_home = True
                team2_home = False
                if conf == "C2C":
                    if lower_seed.team_id == "236":
                        team1_home = False
                        team2_home = True
                    elif higher_seed.team_id != "236":
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

            tournament_teams = standings[: structure.total_teams]
            if len(tournament_teams) == 5 and structure.byes == 3:
                # Special case: In 5-team/3-bye tournaments, 4/5 winner must play 1 seed
                current_team_standings.sort(key=lambda x: standings.index(x))
            elif (
                len(tournament_teams) == 7
                and structure.byes == 1
                and not structure.reseeding
            ):
                # Special case: In 7-team/1-bye tournaments:
                # - 4/5 winner must play 1 seed
                # - 2/7 winner must play 3/6 winner
                if len(remaining_teams) == 4:  # Semifinal round
                    seed_positions = {s.team_id: i for i, s in enumerate(standings)}
                    # Sort by seed to get 1,2,3,4 ordering
                    current_team_standings.sort(key=lambda x: seed_positions[x.team_id])
                    # Reorder to get proper semifinals: 1 vs 4/5 winner, 2/7 winner vs 3/6 winner
                    # We know 4/5 winner will be seed 4 after sort
                    # We want: [1, 2/7 winner, 3/6 winner, 4/5 winner]
                    current_team_standings = [
                        current_team_standings[0],  # 1 seed
                        current_team_standings[1],  # Next highest seed (2/7 winner)
                        current_team_standings[2],  # Next highest seed (3/6 winner)
                        current_team_standings[3],  # 4/5 winner
                    ]
            elif structure.reseeding:
                current_team_standings.sort(key=lambda x: standings.index(x))

            matchups = pair_teams_by_seed(current_team_standings)
            round_name = get_round_name(len(remaining_teams))

            next_round_teams = []
            for higher_seed, lower_seed in matchups:
                team1_home = True
                team2_home = False
                if conf == "C2C":
                    if lower_seed.team_id == "236":
                        team1_home = False
                        team2_home = True
                    elif higher_seed.team_id != "236":
                        team1_home = False
                        team2_home = False
                elif conf == "CCIW":
                    if round_name in ["Final", "Semifinal"]:
                        team1_home = higher_seed.team_id == "160"
                        team2_home = lower_seed.team_id == "160"
                        if not (team1_home or team2_home):
                            team1_home = False
                            team2_home = False
                elif conf == "ODAC":
                    if round_name in ["Final", "Semifinal"]:
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

        # Only if we have teams to process, get the conference champion
        if remaining_teams:
            # The last team remaining should be the champion
            final_team_id = remaining_teams[0]

            # Extra verification with get_conference_champion
            champion_id = get_conference_champion(
                conf, structure.provisional_teams, all_tournament_games
            )

            # Use the champion from get_conference_champion if available, otherwise use the last remaining team
            final_champion_id = (
                champion_id if champion_id is not None else final_team_id
            )

            if final_champion_id:
                conference_champions[conf] = final_champion_id

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
    games, team_conference_data, conference_tournament_info
):
    completed_games = []
    for game in games:
        # First check if game is completed
        team1_score = game.get("team1_score", 0)
        team2_score = game.get("team2_score", 0)
        if team1_score == 0 and team2_score == 0:
            continue  # Skip uncompleted games

        game_date = game.get("date")
        team1_id = game.get("team1_id")
        team2_id = game.get("team2_id")

        team1_conference = team_conference_data.get(team1_id)
        team2_conference = team_conference_data.get(team2_id)

        if not team1_conference or not team2_conference:
            continue

        team1_conf_name = team1_conference.conference
        team2_conf_name = team2_conference.conference

        if team1_conf_name != team2_conf_name:
            continue

        reg_season_end_date = conference_tournament_info.get(team1_conf_name)
        if reg_season_end_date and int(game_date) > int(reg_season_end_date):
            completed_games.append(game)

    return completed_games


def determine_tournament_state(
    completed_games: List[dict],
    tournament_structure: ConferenceTournament,
    standings: List[ConferenceStanding],
) -> Tuple[List[str], List[str], str]:
    """
    Determine the current state of a conference tournament.

    Args:
        completed_games: List of completed tournament games for the conference
        tournament_structure: Structure of the tournament
        standings: Conference standings in seed order

    Returns:
        Tuple of (active_teams, bye_teams, next_game_date)
    """
    conf = tournament_structure.conference

    if not completed_games:
        # Tournament hasn't started
        tournament_teams = standings[: tournament_structure.total_teams]
        bye_teams = [
            team.team_id for team in tournament_teams[: tournament_structure.byes]
        ]
        active_teams = [
            team.team_id for team in tournament_teams[tournament_structure.byes :]
        ]
        return active_teams, bye_teams, "20250302"

    # Sort games chronologically
    completed_games.sort(key=lambda x: x["date"])
    next_game_date = str(int(completed_games[-1]["date"]) + 1).zfill(8)

    # Analyze the completed games to determine what round we're in
    # Count how many games have been played
    num_completed = len(completed_games)
    total_teams = tournament_structure.total_teams
    num_byes = tournament_structure.byes

    # Calculate number of teams in each round
    round_info = get_tournament_rounds(total_teams, num_byes)

    # Find all teams that participated in games
    participating_teams = set()
    for game in completed_games:
        participating_teams.add(game["team1_id"])
        participating_teams.add(game["team2_id"])

    # Find winner of each game
    winners = []
    for game in completed_games:
        winner_id = (
            game["team1_id"]
            if game["team1_score"] > game["team2_score"]
            else game["team2_id"]
        )
        winners.append(winner_id)

    # Get teams with byes that haven't played yet
    tournament_teams = standings[: tournament_structure.total_teams]
    bye_teams = [team.team_id for team in tournament_teams[: tournament_structure.byes]]
    unplayed_bye_teams = [
        team_id for team_id in bye_teams if team_id not in participating_teams
    ]

    # Determine which round we're in based on games played
    expected_games_so_far = 0
    current_round_idx = -1

    for i, (round_name, num_teams, num_games) in enumerate(round_info):
        expected_games_so_far += num_games
        if num_completed <= expected_games_so_far:
            current_round_idx = i
            break

    if current_round_idx == -1:
        # This shouldn't happen, but in case it does, assume we're in the final round
        current_round_idx = len(round_info) - 1

    # Get the current round name and details
    current_round = round_info[current_round_idx]

    # If we've completed exactly the number of games for this round, we move to the next round
    if (
        num_completed == expected_games_so_far
        and current_round_idx < len(round_info) - 1
    ):
        # We've completed this round, move to the next one
        current_round_idx += 1
        current_round = round_info[current_round_idx]

        # The winners of the completed games plus any unplayed bye teams move to the next round
        # Only take the most recent winners to form the next round
        recent_winners = winners[-current_round[1] + len(unplayed_bye_teams) :]
        active_teams = recent_winners + unplayed_bye_teams
    else:
        # We're still in the current round, some games might be finished
        # Calculate how many games have been played in this round
        games_in_this_round = num_completed - (expected_games_so_far - current_round[2])
        games_remaining = current_round[2] - games_in_this_round

        if games_remaining == 0:
            # All games in this round are done, winners move to next round
            winners_from_this_round = winners[-games_in_this_round:]
            active_teams = winners_from_this_round + unplayed_bye_teams
        else:
            # Some games remain in this round
            # Find teams that already played in this round
            teams_in_this_round = set()
            recent_games = completed_games[-games_in_this_round:]
            for game in recent_games:
                teams_in_this_round.add(game["team1_id"])
                teams_in_this_round.add(game["team2_id"])

            # Winners from this round so far
            winners_from_this_round = []
            for game in recent_games:
                winner = (
                    game["team1_id"]
                    if game["team1_score"] > game["team2_score"]
                    else game["team2_id"]
                )
                winners_from_this_round.append(winner)

            # Calculate teams that still need to play in this round
            all_teams_this_round = tournament_teams[
                num_byes : num_byes + current_round[1]
            ]
            all_team_ids_this_round = [team.team_id for team in all_teams_this_round]
            remaining_teams_this_round = [
                team_id
                for team_id in all_team_ids_this_round
                if team_id not in teams_in_this_round
                or team_id in winners_from_this_round
            ]

            # Add any unplayed bye teams
            active_teams = remaining_teams_this_round + unplayed_bye_teams

    return active_teams, [], next_game_date


def get_tournament_rounds(total_teams: int, byes: int) -> List[Tuple[str, int, int]]:
    """
    Calculate the rounds, number of teams, and number of games for a tournament.

    Args:
        total_teams: Total number of teams in the tournament
        byes: Number of teams with first-round byes

    Returns:
        List of tuples (round_name, num_teams, num_games)
    """
    rounds = []

    # Calculate teams in first round (excluding byes)
    first_round_teams = total_teams - byes

    # First round might not exist if all teams have byes
    if first_round_teams > 0:
        num_games = first_round_teams // 2
        round_name = get_round_name(first_round_teams + byes)
        rounds.append((round_name, first_round_teams, num_games))

    # Teams after first round (winners + byes)
    remaining = (first_round_teams // 2) + byes

    # Continue with subsequent rounds
    while remaining > 1:
        num_games = remaining // 2
        round_name = get_round_name(remaining)
        rounds.append((round_name, remaining, num_games))
        remaining = remaining // 2

    return rounds


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
    """
    Get eligible conference champion, accounting for provisional teams.
    Args:
        conference: Conference name
        provisional_teams: Set of team IDs that are ineligible for autobid
        tournament_games: List of all tournament games
    Returns:
        Team ID of the conference champion, or None if no eligible champion
    """
    # Find all games for this conference
    conf_games = [g for g in tournament_games if g.get("conference") == conference]

    # Get all games marked as "Final"
    final_games = [g for g in conf_games if g.get("round") == "Final"]

    if not final_games:
        return None

    # Sort by date to ensure we get the latest final (in case there are multiple)
    final_games.sort(key=lambda x: x.get("date", "00000000"))
    championship_game = final_games[-1]

    # Get team IDs and scores
    team1_id = championship_game.get("team1_id")
    team2_id = championship_game.get("team2_id")
    team1_score = championship_game.get("team1_score")
    team2_score = championship_game.get("team2_score")

    # Determine winner
    winner_id = team1_id if team1_score > team2_score else team2_id

    # Check if winner is eligible
    if winner_id not in provisional_teams:
        return winner_id

    # If winner is provisional, return runner-up if eligible
    runner_up = team2_id if team1_id == winner_id else team1_id
    if runner_up not in provisional_teams:
        return runner_up
    else:
        return None


def calculate_head_to_head(
    season_results: List[dict], team_a: str, team_b: str
) -> Tuple[int, int]:
    """Calculate head-to-head record between two teams."""
    wins_a = 0
    wins_b = 0
    for game in season_results:
        if game["team1_id"] == team_a and game["team2_id"] == team_b:
            if game["team1_score"] > game["team2_score"]:
                wins_a += 1
            else:
                wins_b += 1
        elif game["team1_id"] == team_b and game["team2_id"] == team_a:
            if game["team1_score"] > game["team2_score"]:
                wins_b += 1
            else:
                wins_a += 1
    return (wins_a, wins_b)
