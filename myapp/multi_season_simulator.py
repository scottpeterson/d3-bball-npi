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
from .conf_tournaments import load_tournament_structures
from .bid_thieves import determine_at_large_bid
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
    qf_pool_c_pct: float
    sf_pool_c_pct: float
    f_pool_c_pct: float
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
    exit_round: str
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
        final_teams = main(use_season_results=True)

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
        
         # Get provisional teams from all conferences
        provisional_teams = set()
        tournament_structures = load_tournament_structures(base_path, year)
        for structure in tournament_structures.values():
           provisional_teams.update(structure.provisional_teams)

        # Analyze bid thieves
        bid_thieves, _ = analyze_tournament_bid_thieves(
            tournament_games,
            conference_champions,
            conference_teams,
            final_teams,
            auto_bid_recipients,
            provisional_teams
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
            ]

            # Auto bid status already determined
            got_auto_bid = team_id in auto_bid_recipients

            # Determine at-large bid (C)
            got_at_large = determine_at_large_bid(
               team_id, 
               final_teams, 
               auto_bid_recipients,
               provisional_teams
           )

            results[team_id] = TeamSimResult(
                simulation_number=sim_number,
                team_name=team_name,
                wins=wins,
                losses=losses,
                npi_rank=npi,
                got_auto_bid=got_auto_bid,
                got_at_large=got_at_large,
                made_tournament=got_auto_bid or got_at_large,
            )

            # Get at-large bids for tournament analysis
            at_large_bids = {
                team_id for team_id, result in results.items() 
                if result.got_at_large
            }

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
            else "-"
        )
        sf_pool_c_pct = (
            (tourn_stats.semifinal_pool_c / tourn_stats.semifinal_total * 100)
            if tourn_stats.semifinal_total > 0
            else "-"
        )
        f_pool_c_pct = (
            (tourn_stats.final_pool_c / tourn_stats.final_total * 100)
            if tourn_stats.final_total > 0
            else "-"
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
            rank=0,
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
            
            for team_id, result in sim_results.items():
                all_results[team_id].append(result)
            for team_id in sim_bid_thieves:
                bid_thief_counts[team_id] += 1
            per_sim_bid_count.append(len(sim_bid_thieves))


            # Process tournament results
            for team_id, result in tourn_results.items():
                stats = tournament_stats[team_id]

                # Validate exit_round and got_pool_c
                if result.exit_round not in {'Quarterfinal', 'Semifinal', 'Final'}:
                    print(f"Warning: Unexpected exit_round '{result.exit_round}' for team {team_id}")
                    continue
                    
                if result.got_pool_c is None:
                    print(f"Warning: got_pool_c is None for team {team_id}")
                    continue

                # Update stats based on exit round
                if result.exit_round == 'Final':
                    stats.final_total += 1
                    stats.semifinal_total += 1
                    stats.quarterfinal_total += 1
                    if result.got_pool_c:
                        stats.final_pool_c += 1  # Only increment pool_c for the exit round
                elif result.exit_round == 'Semifinal':
                    stats.semifinal_total += 1
                    stats.quarterfinal_total += 1
                    if result.got_pool_c:
                        stats.semifinal_pool_c += 1  # Only increment pool_c for the exit round
                elif result.exit_round == 'Quarterfinal':
                    stats.quarterfinal_total += 1
                    if result.got_pool_c:
                        stats.quarterfinal_pool_c += 1  # Only increment pool_c for the exit round


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
    )

    # Debug top teams before writing
    print("\nTop 5 teams by NPI:")
    for team_id, team_stats in sorted_teams[:5]:
        team_name = teams_mapping.get(team_id, f"Unknown ({team_id})")
        conference = conference_teams[team_id].conference
        print(f"{team_name} ({conference}): NPI = {team_stats.median_rank:.2f}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Team", "Conf", "MedW", "MedL", "Tourn%", "A%", "C%", "ALWYNI%",
            "F-C%", "SF-C%", "QF-C%", "Med-NPI", "Min", "Max", "Rank"
        ])

        for team_id, team_stats in sorted_teams:
            if team_id not in teams_mapping:
                print(f"Warning: No mapping found for team ID {team_id}")
                continue

            team_name = teams_mapping[team_id]
            conference = conference_teams[team_id].conference

            # Handle formatting for Pool C percentages
            def format_pool_c_pct(pct):
                if isinstance(pct, str):
                    return pct
                return f"{pct:.1f}%"

            writer.writerow([
                team_name,
                conference,
                f"{team_stats.median_wins:.1f}",
                f"{team_stats.median_losses:.1f}",
                f"{team_stats.tournament_pct:.1f}%",
                f"{team_stats.auto_bid_pct:.1f}%",
                f"{team_stats.at_large_pct:.1f}%",
                f"{team_stats.alwyni:.1f}%",
                format_pool_c_pct(team_stats.f_pool_c_pct),
                format_pool_c_pct(team_stats.sf_pool_c_pct),
                format_pool_c_pct(team_stats.qf_pool_c_pct),
                f"{team_stats.median_rank:.1f}",
                f"{team_stats.min_rank:.1f}",
                f"{team_stats.max_rank:.1f}",
                team_stats.rank,
            ])


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
    
    # Filter for tournament games (assuming tournament games are after a specific date)
    tournament_games = [game for game in season_results if game["date"] >= "20250302"]
    
    # Group tournament games by conference
    conference_games = defaultdict(list)
    for game in tournament_games:
        team1_conf = conference_teams[game["team1_id"]].conference
        team2_conf = conference_teams[game["team2_id"]].conference
        
        # Skip UAA conference games
        if team1_conf == "UAA" or team2_conf == "UAA":
            continue
        
        # Only consider games within the same conference
        if team1_conf == team2_conf:
            conference_games[team1_conf].append(game)
    
    # Find the championship game for each conference
    for conf, conf_games in conference_games.items():
        # Sort games by date to get the last (championship) game
        conf_games.sort(key=lambda x: x["date"], reverse=True)
        
        if conf_games:
            # Take the first game after sorting (latest game)
            championship_game = conf_games[0]
            
            # Determine the winner
            winner_id = (
                championship_game["team1_id"]
                if championship_game["team1_score"] > championship_game["team2_score"]
                else championship_game["team2_id"]
            )
            
            champions[winner_id] = conf
    
    return champions


def get_conference_tournament_results(
    tournament_games: List[dict],
    conference_teams: Dict[str, ConferenceTeam],
    conference_champions: Set[str],
    at_large_bids: Set[str]
) -> Dict[str, TeamTournamentResult]:
    """Analyze conference tournament performance and Pool C bid correlation."""
    results: Dict[str, TeamTournamentResult] = {}
    
    # First group games by conference
    conference_games: Dict[str, List[dict]] = {}
    for game in tournament_games:
        team1_id = game['team1_id']
        team2_id = game['team2_id']
        if team1_id not in conference_teams or team2_id not in conference_teams:
            print(f"DEBUG: Skipping game due to missing team: {team1_id} vs {team2_id}")
            continue
            
        conf = conference_teams[team1_id].conference
        if conf not in conference_games:
            conference_games[conf] = []
        conference_games[conf].append(game)
    
    # Process each conference tournament
    for conf, games in conference_games.items():

        # Sort games by date and count games per date
        games_by_date = {}
        for game in sorted(games, key=lambda x: x['date']):
            date = game['date']
            if date not in games_by_date:
                games_by_date[date] = []
            games_by_date[date].append(game)
        
        dates = sorted(games_by_date.keys())
        games_per_date = {date: len(games_by_date[date]) for date in dates}
        
        # Map each date to a round
        round_by_date = {}
        remaining_dates = sorted(dates)
        
        # Find quarterfinal date
        max_games = max(games_per_date.values())
        qf_games = [d for d in dates if games_per_date[d] >= 4]
        if qf_games:
            qf_date = min(qf_games)
            round_by_date[qf_date] = 'Quarterfinal'
            remaining_dates.remove(qf_date)
        
        # Find final date
        final_candidates = [d for d in remaining_dates if games_per_date[d] == 1]
        if final_candidates:
            final_date = max(final_candidates)
            round_by_date[final_date] = 'Final'
            if final_date in remaining_dates:
                remaining_dates.remove(final_date)
        
        # Assign semifinals
        for date in remaining_dates:
            round_by_date[date] = 'Semifinal'
                    
        # Process each game
        eliminated_teams = set()
        for date in dates:
            if date not in round_by_date:
                print(f"WARNING: Date {date} has no round assignment")
                continue
                
            round_name = round_by_date[date]
            for game in games_by_date[date]:
                team1_id = game['team1_id'] 
                team2_id = game['team2_id']
                team1_score = game['team1_score']
                team2_score = game['team2_score']
                
                team1_is_winner = team1_score > team2_score
                winner_id = team1_id if team1_is_winner else team2_id
                loser_id = team2_id if team1_is_winner else team1_id
                
                if loser_id not in eliminated_teams:
                    eliminated_teams.add(loser_id)
                    results[loser_id] = TeamTournamentResult(
                        team_id=loser_id,
                        conference=conf,
                        exit_round=round_name,
                        got_pool_c=loser_id in at_large_bids
                    )
                
                if round_name == 'Final':
                    # Remove recording of winner - only record the loser
                    if loser_id not in eliminated_teams:
                        eliminated_teams.add(loser_id)
                        results[loser_id] = TeamTournamentResult(
                            team_id=loser_id,
                            conference=conf,
                            exit_round=round_name,
                            got_pool_c=loser_id in at_large_bids
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