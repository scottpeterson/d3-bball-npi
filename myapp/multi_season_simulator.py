from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import csv
import statistics
import time
from collections import defaultdict
from .load_teams import load_teams
from .simulation import load_efficiency_data, simulate_full_season, load_conference_data, load_tournament_structures
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
    got_auto_bid: bool  # A
    got_at_large: bool  # C
    made_tournament: bool  # in/out

@dataclass
class TeamStats:
    median_wins: float
    median_losses: float
    auto_bid_pct: float
    at_large_pct: float
    tournament_pct: float
    median_rank: float    # This is actually storing the NPI value
    min_rank: float      # Change to float since it's NPI
    max_rank: float      # Change to float since it's NPI

@dataclass
class TeamSimResult:
    simulation_number: int
    team_name: str
    wins: int
    losses: int
    npi_rank: int
    got_auto_bid: bool  # A
    got_at_large: bool  # C
    made_tournament: bool  # in/out

def run_single_simulation(base_path: Path, year: str, sim_number: int) -> Dict[str, TeamSimResult]:
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
                    season_results.append({
                        'date': cols[1].strip(),
                        'team1_id': cols[2].strip(),
                        'team2_id': cols[5].strip(),
                        'team1_score': int(cols[4]),
                        'team2_score': int(cols[7])
                    })
        
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
            elif (conference_teams[team_id].conference == "UAA" and
                  is_uaa_regular_season_champion(team_id, season_results, conference_teams)):
                auto_bid_recipients.add(team_id)
        
        # Now process all teams for complete results
        for team_id, team_stats in final_teams.items():
            if not team_stats["has_games"]:
                continue
                
            team_name = valid_teams[team_id]
            wins = team_stats["wins"]
            losses = team_stats["losses"]
            npi = team_stats["npi"]  # This is what's actually in the team_stats dictionary
            
            # Auto bid status already determined
            got_auto_bid = team_id in auto_bid_recipients
            
            # Determine at-large bid (C)
            got_at_large = determine_at_large_bid(team_id, final_teams, auto_bid_recipients)
            
            results[team_id] = TeamSimResult(
                simulation_number=sim_number,
                team_name=team_name,
                wins=wins,
                losses=losses,
                npi_rank=npi,  # Using the NPI value directly
                got_auto_bid=got_auto_bid,
                got_at_large=got_at_large,
                made_tournament=got_auto_bid or got_at_large
            )
        
        return results
            
    except Exception as e:
        print(f"Error in simulation {sim_number}: {e}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return {}

def calculate_team_stats(all_results: Dict[str, List[TeamSimResult]]) -> Dict[str, TeamStats]:
    """Calculate statistics across all simulations for each team."""
    team_stats = {}
    
    print(f"\nProcessing results for {len(all_results)} teams")  # Debug print
    
    for team_id, results in all_results.items():
        if not results:
            print(f"No results for team {team_id}")  # Debug print
            continue
                        
        wins = [r.wins for r in results]
        losses = [r.losses for r in results]
        ranks = [r.npi_rank for r in results]
        auto_bids = sum(1 for r in results if r.got_auto_bid)
        at_large_bids = sum(1 for r in results if r.got_at_large)
        tournament_appearances = sum(1 for r in results if r.made_tournament)
        total_sims = len(results)
        
        team_stats[team_id] = TeamStats(
            median_wins=statistics.median(wins),
            median_losses=statistics.median(losses),
            auto_bid_pct=(auto_bids / total_sims) * 100,
            at_large_pct=(at_large_bids / total_sims) * 100,
            tournament_pct=(tournament_appearances / total_sims) * 100,
            median_rank=statistics.median(ranks),
            min_rank=min(ranks),
            max_rank=max(ranks)
        )
    
    print(f"\nCalculated stats for {len(team_stats)} teams")  # Debug print
    return team_stats

def run_multiple_simulations(base_path: Path, year: str, num_sims: int = 1000) -> Dict[str, TeamStats]:
    """Run multiple season simulations and compile statistics."""
    all_results = defaultdict(list)
    total_start_time = time.time()
    
    for sim_number in range(1, num_sims + 1):
        try:
            sim_start_time = time.time()
            print(f"\nStarting simulation {sim_number}")
            
            # Add progress indicators for each major step
            print("  Running season simulation...", flush=True)
            sim_results = run_single_simulation(base_path, year, sim_number)
            
            print("  Processing results...", flush=True)
            # Collect results for each team
            for team_id, result in sim_results.items():
                all_results[team_id].append(result)
                
            sim_end_time = time.time()
            sim_duration = sim_end_time - sim_start_time
            print(f"Completed simulation {sim_number} in {sim_duration:.2f} seconds")
            
        except Exception as e:
            print(f"Error in simulation {sim_number}:")
            print(f"  {str(e)}")
            print("Continuing with next simulation...")
            continue
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nAll simulations completed in {total_duration:.2f} seconds")
    print(f"Average time per simulation: {total_duration/num_sims:.2f} seconds")
    
    # Calculate final statistics
    return calculate_team_stats(all_results)

import csv

import csv

def save_simulation_stats(stats: Dict[str, TeamStats], base_path: Path, year: str, conference_teams: Dict[str, ConferenceTeam]):
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
    sorted_teams = sorted(stats.items(),
                         key=lambda x: x[1].median_rank,
                         reverse=True)  # Higher NPI is better
    
    # Debug top teams before writing
    print("\nTop 5 teams by NPI:")
    for team_id, team_stats in sorted_teams[:5]:
        team_name = teams_mapping.get(team_id, f"Unknown ({team_id})")
        conference = conference_teams[team_id].conference
        print(f"{team_name} ({conference}): NPI = {team_stats.median_rank:.2f}")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Team', 'Conf', 'Med-W', 'Med-L', 'Auto Bid %',
            'At-Large %', 'Tourn %', 'Med-NPI', 'Min', 'Max'
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
                f"{team_stats.tournament_pct:.1f}%",
                f"{team_stats.median_rank:.1f}",
                team_stats.min_rank,
                team_stats.max_rank
            ])

def get_uaa_standings(season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]) -> List[str]:
    """Calculate UAA standings based on overall record."""
    uaa_records = {}
    
    # Find UAA teams and initialize their records
    for team_id, team in conference_teams.items():
        if team.conference == "UAA":
            uaa_records[team_id] = {"wins": 0, "losses": 0}
    
    # Calculate records
    for game in season_results:
        team1_id = game['team1_id']
        team2_id = game['team2_id']
        
        # Only process if both teams are UAA
        if (team1_id in uaa_records and team2_id in uaa_records):
            if game['team1_score'] > game['team2_score']:
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
        reverse=True
    )
    
    return sorted_teams

def is_uaa_regular_season_champion(team_id: str, season_results: List[dict], 
                                 conference_teams: Dict[str, ConferenceTeam]) -> bool:
    """Determine if a team won the UAA regular season."""
    standings = get_uaa_standings(season_results, conference_teams)
    return len(standings) > 0 and standings[0] == team_id

from collections import defaultdict
from typing import List, Tuple

def find_latest_conference_games(season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]) -> List[Tuple[str, str]]:
    """
    Find the latest date of games between teams from the same conference.
    
    Args:
        season_results: List of game results.
        conference_teams: Dictionary mapping team IDs to their conference.
        
    Returns:
        List of tuples containing the conference name and the latest game date.
    """
    conference_game_dates = defaultdict(str)
    
    for game in season_results:
        team1_conf = conference_teams[game['team1_id']].conference
        team2_conf = conference_teams[game['team2_id']].conference
        
        # Skip games where either team is in the UAA conference
        if team1_conf == 'UAA' or team2_conf == 'UAA':
            continue
        
        # If the teams are in the same conference, update the latest date for that conference
        if team1_conf == team2_conf:
            game_date = game['date']
            if game_date > conference_game_dates[team1_conf]:
                conference_game_dates[team1_conf] = game_date
    
    # Convert the dictionary to a list of (conference, latest_date) tuples
    return list(conference_game_dates.items())

def get_conference_champions(season_results: List[dict], conference_teams: Dict[str, ConferenceTeam]) -> Dict[str, str]:
    """Extract conference tournament champions and their conferences."""
    champions = {}

    # 1. Load all the games in season_results
    all_games = season_results

    # 2. Group the games by conference, excluding UAA
    conference_games = defaultdict(list)
    for game in all_games:
        team1_conf = conference_teams[game['team1_id']].conference
        team2_conf = conference_teams[game['team2_id']].conference
        
        # Skip games where either team is in the UAA conference
        if team1_conf == 'UAA' or team2_conf == 'UAA':
            continue
        
        # Add the game to the appropriate conference's list
        conference_games[team1_conf].append(game)
        conference_games[team2_conf].append(game)

    # 3. For each conference, find the game with the latest date where 2 teams from the same conference faced each other
    for conf, conf_games in conference_games.items():
        # Sort the games by date in descending order
        conf_games.sort(key=lambda x: x['date'], reverse=True)
        
        # Find the first game where the teams are in the same conference
        for game in conf_games:
            team1_conf = conference_teams[game['team1_id']].conference
            team2_conf = conference_teams[game['team2_id']].conference
            if team1_conf == team2_conf:
                winner_id = game['team1_id'] if game['team1_score'] > game['team2_score'] else game['team2_id']
                champions[winner_id] = team1_conf
                break
    
    return champions

def determine_at_large_bid(team_id: str, 
                         all_teams: Dict[str, Dict],
                         auto_bid_recipients: Set[str]) -> bool:
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
        {'team_id': tid, 'npi': stats['npi']}
        for tid, stats in all_teams.items()
        if tid not in auto_bid_recipients and stats.get("has_games", False)
    ]
    
    # Sort the teams by NPI in descending order (higher NPI is better)
    non_auto_bid_teams.sort(key=lambda x: x['npi'], reverse=True)
    
    # Take the top 21 teams for at-large bids
    at_large_bids = {team['team_id'] for team in non_auto_bid_teams[:21]}
    
    return team_id in at_large_bids