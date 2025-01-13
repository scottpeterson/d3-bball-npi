from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Set
from pathlib import Path
import csv
import statistics
import time
from collections import defaultdict
from .load_teams import load_teams
from .bid_thieves import analyze_tournament_bid_thieves 
from .simulation import (
    load_efficiency_data,
    simulate_full_season,
    load_conference_data,
)
from .main import main


@dataclass
class ConferenceTeam:
    """Class for storing team conference information."""

    team_id: str
    team_name: str
    conference: str


@dataclass
class TeamSimResult:
    simulation_number: int
    team_name: str
    wins: int
    losses: int
    npi_rank: int
    got_auto_bid: bool
    got_at_large: bool
    made_tournament: bool


@dataclass
class TeamStats:
    median_wins: float
    median_losses: float
    auto_bid_pct: float
    at_large_pct: float
    alwyni: float
    tournament_pct: float
    qf_pool_c_pct: float  # New field for Pool C % after QF loss
    sf_pool_c_pct: float  # New field for Pool C % after SF loss
    f_pool_c_pct: float   # New field for Pool C % after F loss
    median_rank: float
    min_rank: float
    max_rank: float
    rank: int

@dataclass
class TeamSimResult:
    simulation_number: int
    team_name: str
    wins: int
    losses: int
    npi_rank: int
    got_auto_bid: bool
    got_at_large: bool
    made_tournament: bool

@dataclass
class TeamTournamentResult:
    team_id: str
    conference: str
    exit_round: str  # 'quarterfinal', 'semifinal', 'final', or 'champion'
    got_pool_c: bool

@dataclass
class ConferenceTournamentStats:
    quarterfinal_total: int = 0
    quarterfinal_pool_c: int = 0
    semifinal_total: int = 0
    semifinal_pool_c: int = 0
    final_total: int = 0
    final_pool_c: int = 0

def run_single_simulation(
    base_path: Path, year: str, sim_number: int
) -> Tuple[Dict[str, TeamSimResult], List[str], Dict[str, TeamTournamentResult]]:
    """Run a single season simulation and process results."""
    results = {}

    try:
        # Load necessary data
        print(f"\nSimulation {sim_number}:")
        valid_teams = load_teams(base_path, year)
        team_data = load_efficiency_data(base_path, int(year))

        # Run full season simulation
        simulate_full_season(base_path, year, valid_teams, team_data)

        # Load season results
        season_results = []
        with open(base_path / year / "season_results.txt", "r") as file:
            for line in file:
                cols = line.strip().split(",")
                if len(cols) >= 8:
                    season_results.append(
                        {
                            "date": cols[1].strip(),
                            "team1_id": cols[2].strip(),
                            "team2_id": cols[5].strip(),
                            "team1_score": int(cols[4]),
                            "team2_score": int(cols[7]),
                        }
                    )

        # Run NPI calculations
        final_teams = main()

        if final_teams is None:
            raise ValueError("NPI calculations failed to return results")

        # Get conference champions (both tournament and regular season for UAA)
        conference_teams = load_conference_data(base_path, year)

        # Get all auto-bid recipients first
        auto_bid_recipients = set()

        # Process NPI results for each team to find auto bids first
        for team_id, team_stats in final_teams.items():
            if not team_stats["has_games"]:
                continue

            # Determine auto bid (A)
            if team_id in get_conference_champions(season_results, conference_teams):
                auto_bid_recipients.add(team_id)
            elif conference_teams[
                team_id
            ].conference == "UAA" and is_uaa_regular_season_champion(
                team_id, season_results, conference_teams
            ):
                auto_bid_recipients.add(team_id)

        # Filter tournament games from season_results
        tournament_games = [game for game in season_results if game["date"] >= "20250302"]
        conference_champions = get_conference_champions(season_results, conference_teams)
        
        # Analyze bid thieves
        bid_thieves, _ = analyze_tournament_bid_thieves(
            tournament_games,
            conference_champions,
            conference_teams,
            final_teams,
            auto_bid_recipients
        )
        # Now process all teams for complete results
        for team_id, team_stats in final_teams.items():
            if not team_stats["has_games"]:
                continue

            team_name = valid_teams[team_id]
            wins = team_stats["wins"]
            losses = team_stats["losses"]
            npi = team_stats[
                "npi"
            ]  # This is what's actually in the team_stats dictionary

            # Auto bid status already determined
            got_auto_bid = team_id in auto_bid_recipients

            # Determine at-large bid (C)
            got_at_large = determine_at_large_bid(
                team_id, final_teams, auto_bid_recipients
            )

            results[team_id] = TeamSimResult(
                simulation_number=sim_number,
                team_name=team_name,
                wins=wins,
                losses=losses,
                npi_rank=npi,  # Using the NPI value directly
                got_auto_bid=got_auto_bid,
                got_at_large=got_at_large,
                made_tournament=got_auto_bid or got_at_large,
            )

            # Get at-large bids for tournament analysis
            at_large_bids = {
                team_id for team_id, result in results.items() 
                if result.got_at_large
            }
        
            # Get tournament results
            tournament_results = get_conference_tournament_results(
                tournament_games,
                conference_teams,
                conference_champions,
                at_large_bids
            )

        return results, bid_thieves, tournament_results

    except Exception as e:
        print(f"Error in simulation {sim_number}: {e}")
        print("Traceback:")
        import traceback

        traceback.print_exc()
        return {}


def calculate_team_stats(
    all_results: Dict[str, List[TeamSimResult]],
    tournament_stats: Dict[str, ConferenceTournamentStats]
) -> Dict[str, TeamStats]:
    """Calculate statistics across all simulations for each team."""
    team_stats = {}
    print(f"\nProcessing results for {len(all_results)} teams")

    # First pass - calculate all basic stats
    for team_id, results in all_results.items():
        if not results:
            print(f"No results for team {team_id}")
            continue

        wins = [r.wins for r in results]
        losses = [r.losses for r in results]
        ranks = [r.npi_rank for r in results]
        auto_bids = sum(1 for r in results if r.got_auto_bid)
        at_large_bids = sum(1 for r in results if r.got_at_large)
        tournament_appearances = sum(1 for r in results if r.made_tournament)
        total_sims = len(results)

        # Calculate ALWYNI (At-Large When You Need It)
        non_auto_bid_sims = sum(1 for r in results if not r.got_auto_bid)
        alwyni = (
            (at_large_bids / non_auto_bid_sims * 100) if non_auto_bid_sims > 0 else 0
        )

        # Get tournament stats for this team
        tourn_stats = tournament_stats.get(team_id, ConferenceTournamentStats())
        
        # Calculate Pool C percentages for each round
        qf_pool_c_pct = (
            (tourn_stats.quarterfinal_pool_c / tourn_stats.quarterfinal_total * 100)
            if tourn_stats.quarterfinal_total > 0
            else 0.0
        )
        sf_pool_c_pct = (
            (tourn_stats.semifinal_pool_c / tourn_stats.semifinal_total * 100)
            if tourn_stats.semifinal_total > 0
            else 0.0
        )
        f_pool_c_pct = (
            (tourn_stats.final_pool_c / tourn_stats.final_total * 100)
            if tourn_stats.final_total > 0
            else 0.0
        )

        team_stats[team_id] = TeamStats(
            median_wins=statistics.median(wins),
            median_losses=statistics.median(losses),
            auto_bid_pct=(auto_bids / total_sims) * 100,
            at_large_pct=(at_large_bids / total_sims) * 100,
            tournament_pct=(tournament_appearances / total_sims) * 100,
            alwyni=alwyni,
            median_rank=statistics.median(ranks),
            min_rank=min(ranks),
            max_rank=max(ranks),
            rank=0,  # temporary placeholder
            qf_pool_c_pct=qf_pool_c_pct,
            sf_pool_c_pct=sf_pool_c_pct,
            f_pool_c_pct=f_pool_c_pct
        )

    # Second pass - assign ranks based on tournament_pct
    # Sort teams by tournament_pct in descending order
    sorted_teams = sorted(
        team_stats.items(), key=lambda x: x[1].tournament_pct, reverse=True
    )

    # Assign ranks (1-based ranking)
    for rank, (team_id, stats) in enumerate(sorted_teams, 1):
        team_stats[team_id] = replace(stats, rank=rank)

    print(f"\nCalculated stats for {len(team_stats)} teams")
    return team_stats


def run_multiple_simulations(
    base_path: Path, year: str, num_sims: int = 1000
) -> Tuple[Dict[str, TeamStats], Dict[str, int], List[int], Dict[str, ConferenceTournamentStats]]:
    all_results = defaultdict(list)
    bid_thief_counts = defaultdict(int)
    per_sim_bid_count = []
    tournament_stats = defaultdict(ConferenceTournamentStats)
    total_start_time = time.time()

    for sim_number in range(1, num_sims + 1):
        try:
            sim_start_time = time.time()
            print(f"\nStarting simulation {sim_number}")
            print("  Running season simulation...", flush=True)
            
            sim_results, sim_bid_thieves, tourn_results = run_single_simulation(
                base_path, year, sim_number
            )
            print("  Processing results...", flush=True)
            
            for team_id, result in sim_results.items():
                all_results[team_id].append(result)
            for team_id in sim_bid_thieves:
                bid_thief_counts[team_id] += 1
            per_sim_bid_count.append(len(sim_bid_thieves))

            # Process tournament results
            for team_id, result in tourn_results.items():
                stats = tournament_stats[team_id]
                if result.exit_round == 'Quarterfinal':
                    stats.quarterfinal_total += 1
                    if result.got_pool_c:
                        stats.quarterfinal_pool_c += 1
                elif result.exit_round == 'Semifinal':
                    stats.semifinal_total += 1
                    if result.got_pool_c:
                        stats.semifinal_pool_c += 1
                elif result.exit_round == 'Final':
                    stats.final_total += 1
                    if result.got_pool_c:
                        stats.final_pool_c += 1

            sim_duration = time.time() - sim_start_time
            print(f"Completed simulation {sim_number} in {sim_duration:.2f} seconds")

        except Exception as e:
            print(f"Error in simulation {sim_number}:")
            print(f"  {str(e)}")
            continue

    total_duration = time.time() - total_start_time
    print(f"\nAll simulations completed in {total_duration:.2f} seconds")
    print(f"Average time per simulation: {total_duration/num_sims:.2f} seconds")

    team_stats = calculate_team_stats(all_results, tournament_stats)
    return team_stats, bid_thief_counts, per_sim_bid_count, tournament_stats


def save_simulation_stats(
    stats: Dict[str, TeamStats],
    base_path: Path,
    year: str,
    conference_teams: Dict[str, ConferenceTeam],
):
    """Save simulation statistics to CSV with team names and conferences."""
    output_path = base_path / year / "simulation_stats.csv"

    # Load team mappings
    teams_mapping = {}
    mapping_path = base_path / year / "teams_mapping.txt"
    try:
        with open(mapping_path, "r") as file:
            next(file)  # Skip header
            for line in file:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    team_id = parts[0].strip()
                    scott_name = parts[2].strip()
                    teams_mapping[team_id] = scott_name
    except Exception as e:
        print(f"Error loading team mappings: {e}")
        return

    # Sort by median rank (since we're storing NPIs in median_rank field)
    sorted_teams = sorted(
        stats.items(), key=lambda x: x[1].median_rank, reverse=True
    )  # Higher NPI is better

    # Debug top teams before writing
    print("\nTop 5 teams by NPI:")
    for team_id, team_stats in sorted_teams[:5]:
        team_name = teams_mapping.get(team_id, f"Unknown ({team_id})")
        conference = conference_teams[team_id].conference
        print(f"{team_name} ({conference}): NPI = {team_stats.median_rank:.2f}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Team",
            "Conf",
            "MedW",
            "MedL",
            "A%",
            "C%",
            "ALWYNI%",
            "Tourn%",
            "QF-C%",
            "SF-C%",
            "F-C%",
            "Med-NPI",
            "Min",
            "Max",
            "Rank",
        ])

        for team_id, team_stats in sorted_teams:
            if team_id not in teams_mapping:
                print(f"Warning: No mapping found for team ID {team_id}")
                continue

            team_name = teams_mapping[team_id]
            conference = conference_teams[team_id].conference
            writer.writerow([
                team_name,
                conference,
                f"{team_stats.median_wins:.1f}",
                f"{team_stats.median_losses:.1f}",
                f"{team_stats.auto_bid_pct:.1f}%",
                f"{team_stats.at_large_pct:.1f}%",
                f"{team_stats.alwyni:.1f}%",
                f"{team_stats.tournament_pct:.1f}%",
                f"{team_stats.qf_pool_c_pct:.1f}%",
                f"{team_stats.sf_pool_c_pct:.1f}%",
                f"{team_stats.f_pool_c_pct:.1f}%",
                f"{team_stats.median_rank:.1f}",
                f"{team_stats.min_rank:.1f}",
                f"{team_stats.max_rank:.1f}",
                team_stats.rank,
            ])
        writer = csv.writer(f)
        writer.writerow(
            [
                "Team",
                "Conf",
                "MedW",
                "MedL",
                "A%",
                "C%",
                "ALWYNI%",
                "Tourn%",
                "QF-C%",  # New column for Pool C % after QF loss
                "SF-C%",  # New column for Pool C % after SF loss
                "F-C%",   # New column for Pool C % after F loss
                "Med-NPI",
                "Min",
                "Max",
                "Rank",
            ]
        )

        for team_id, team_stats in sorted_teams:
            if team_id not in teams_mapping:
                print(f"Warning: No mapping found for team ID {team_id}")
                continue

            team_name = teams_mapping[team_id]
            conference = conference_teams[team_id].conference
            writer.writerow(
                [
                    team_name,
                    conference,
                    f"{team_stats.median_wins:.1f}",
                    f"{team_stats.median_losses:.1f}",
                    f"{team_stats.auto_bid_pct:.1f}%",
                    f"{team_stats.at_large_pct:.1f}%",
                    f"{team_stats.alwyni:.1f}%",
                    f"{team_stats.tournament_pct:.1f}%",
                    f"{team_stats.median_rank:.1f}",
                    f"{team_stats.min_rank:.1f}",
                    f"{team_stats.max_rank:.1f}",
                    team_stats.rank,
                    f"{team_stats.qf_pool_c_pct:.1f}%",
                    f"{team_stats.sf_pool_c_pct:.1f}%",
                    f"{team_stats.f_pool_c_pct:.1f}%",
                ]
            )


def get_uaa_standings(
    season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]
) -> List[str]:
    """Calculate UAA standings based on overall record."""
    uaa_records = {}

    # Find UAA teams and initialize their records
    for team_id, team in conference_teams.items():
        if team.conference == "UAA":
            uaa_records[team_id] = {"wins": 0, "losses": 0}

    # Calculate records
    for game in season_results:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]

        # Only process if both teams are UAA
        if team1_id in uaa_records and team2_id in uaa_records:
            if game["team1_score"] > game["team2_score"]:
                uaa_records[team1_id]["wins"] += 1
                uaa_records[team2_id]["losses"] += 1
            else:
                uaa_records[team2_id]["wins"] += 1
                uaa_records[team1_id]["losses"] += 1

    # Sort by winning percentage
    sorted_teams = sorted(
        uaa_records.keys(),
        key=lambda x: (
            uaa_records[x]["wins"] / (uaa_records[x]["wins"] + uaa_records[x]["losses"])
            if (uaa_records[x]["wins"] + uaa_records[x]["losses"]) > 0
            else 0.0
        ),
        reverse=True,
    )

    return sorted_teams


def is_uaa_regular_season_champion(
    team_id: str,
    season_results: List[dict],
    conference_teams: Dict[str, ConferenceTeam],
) -> bool:
    """Determine if a team won the UAA regular season."""
    standings = get_uaa_standings(season_results, conference_teams)
    return len(standings) > 0 and standings[0] == team_id


def get_conference_champions(
    season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]
) -> Dict[str, str]:
    """Extract conference tournament champions and their conferences."""
    champions = {}

    # 1. Load all the games in season_results
    all_games = season_results

    # 2. Group the games by conference, excluding UAA
    conference_games = defaultdict(list)
    for game in all_games:
        team1_conf = conference_teams[game["team1_id"]].conference
        team2_conf = conference_teams[game["team2_id"]].conference

        # Skip games where either team is in the UAA conference
        if team1_conf == "UAA" or team2_conf == "UAA":
            continue

        # Add the game to the appropriate conference's list
        conference_games[team1_conf].append(game)
        conference_games[team2_conf].append(game)

    # 3. For each conference, find the game with the latest date where 2 teams from the same conference faced each other
    for conf, conf_games in conference_games.items():
        # Sort the games by date in descending order
        conf_games.sort(key=lambda x: x["date"], reverse=True)

        # Find the first game where the teams are in the same conference
        for game in conf_games:
            team1_conf = conference_teams[game["team1_id"]].conference
            team2_conf = conference_teams[game["team2_id"]].conference
            if team1_conf == team2_conf:
                winner_id = (
                    game["team1_id"]
                    if game["team1_score"] > game["team2_score"]
                    else game["team2_id"]
                )
                champions[winner_id] = team1_conf
                break

    return champions


def determine_at_large_bid(
    team_id: str, all_teams: Dict[str, Dict], auto_bid_recipients: Set[str]
) -> bool:
    """
    Determine if a team receives an at-large bid.

    Args:
        team_id: The team being evaluated
        all_teams: Dictionary of all teams and their stats from NPI calculation
        auto_bid_recipients: Set of team IDs that received automatic bids

    Returns:
        bool: True if team receives an at-large bid, False otherwise
    """
    # If team already has an auto bid, they don't need an at-large
    if team_id in auto_bid_recipients:
        return False

    # Get NPI values for all teams without automatic bids
    non_auto_bid_teams = [
        {"team_id": tid, "npi": stats["npi"]}
        for tid, stats in all_teams.items()
        if tid not in auto_bid_recipients and stats.get("has_games", False)
    ]

    # Sort the teams by NPI in descending order (higher NPI is better)
    non_auto_bid_teams.sort(key=lambda x: x["npi"], reverse=True)

    # Take the top 21 teams for at-large bids
    at_large_bids = {team["team_id"] for team in non_auto_bid_teams[:21]}

    return team_id in at_large_bids

def get_conference_tournament_results(
    tournament_games: List[dict],
    conference_teams: Dict[str, ConferenceTeam],
    conference_champions: Set[str],
    at_large_bids: Set[str]
) -> Dict[str, TeamTournamentResult]:
    """
    Analyze conference tournament performance and Pool C bid correlation.
    Determines round based on date ordering since round info isn't in game data.
    """
    # Track results for each team
    results: Dict[str, TeamTournamentResult] = {}
    
    # Group games by conference
    conference_games: Dict[str, List[dict]] = {}
    for game in tournament_games:
        team1_id = game['team1_id']
        team2_id = game['team2_id']
        
        if team1_id not in conference_teams or team2_id not in conference_teams:
            continue
            
        conf = conference_teams[team1_id].conference
        if conf not in conference_games:
            conference_games[conf] = []
        conference_games[conf].append(game)
    
    # Process each conference's tournament
    for conf, games in conference_games.items():
        # Sort games chronologically 
        sorted_games = sorted(games, key=lambda x: x['date'])
        
        # Determine number of rounds based on game count
        num_games = len(sorted_games)
        if num_games >= 7:  # Traditional bracket with quarterfinals
            rounds = ['Quarterfinal'] * 4 + ['Semifinal'] * 2 + ['Final']
        elif num_games >= 3:  # Semifinals only
            rounds = ['Semifinal'] * 2 + ['Final']
        else:  # Championship only
            rounds = ['Final']
            
        # Track teams eliminated in each round
        eliminated_teams = set()
        for game_idx, game in enumerate(sorted_games):
            if game_idx >= len(rounds):  # Skip if we've run out of round names
                continue
                
            round_name = rounds[game_idx]
            team1_id = game['team1_id']
            team2_id = game['team2_id']
            
            # Determine winner and loser
            loser_id = team1_id if game['team1_score'] < game['team2_score'] else team2_id
            winner_id = team2_id if game['team1_score'] < game['team2_score'] else team1_id
            
            # Record result for losing team if not already eliminated
            if loser_id not in eliminated_teams:
                eliminated_teams.add(loser_id)
                results[loser_id] = TeamTournamentResult(
                    team_id=loser_id,
                    conference=conf,
                    exit_round=round_name,
                    got_pool_c=loser_id in at_large_bids
                )
            
            # If this is the final game, record champion
            if round_name == 'Final':
                results[winner_id] = TeamTournamentResult(
                    team_id=winner_id,
                    conference=conf,
                    exit_round='Final',
                    got_pool_c=winner_id in at_large_bids
                )

    return results

def analyze_pool_c_by_round(tournament_results: Dict[str, TeamTournamentResult]) -> Dict[str, Dict[str, int]]:
    """
    Analyze Pool C bid distribution by conference tournament round.
    """
    stats = {
        'Quarterfinal': {'total': 0, 'pool_c': 0},
        'Semifinal': {'total': 0, 'pool_c': 0},
        'Final': {'total': 0, 'pool_c': 0}
    }
    
    for result in tournament_results.values():
        if result.exit_round in stats:
            stats[result.exit_round]['total'] += 1
            if result.got_pool_c:
                stats[result.exit_round]['pool_c'] += 1
    
    return stats