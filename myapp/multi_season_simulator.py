import csv
import statistics
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .bid_thieves import analyze_tournament_bid_thieves, determine_at_large_bid
from .conf_tournaments import ConferenceTournament, load_tournament_structures
from .load_teams import load_teams
from .main import main
from .simulation import load_conference_data, load_efficiency_data, simulate_full_season


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
    made_top_4: bool = False
    made_top_8: bool = False
    made_top_16: bool = False


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
    top_4_pct: float
    top_8_pct: float
    top_16_pct: float


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


@dataclass
class ConferenceGameResult:
    simulation_number: int
    conference: str
    round: str  # e.g., "Semifinal", "Final"
    team1_id: str
    team2_id: str
    team1_score: int
    team2_score: int
    winner_id: str


def run_single_simulation(base_path: Path, year: str, sim_number: int) -> Tuple[
    Dict[str, TeamSimResult],
    Set[str],
    Dict[str, TeamTournamentResult],
    List[ConferenceGameResult],
]:
    """Run a single season simulation and process results."""
    results = {}
    conference_games = []

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

        # Get conference champions
        conference_teams = load_conference_data(base_path, year)
        conference_champions = get_conference_champions(
            season_results, conference_teams
        )

        # Process tournament games
        tournament_games = [
            game for game in season_results if game["date"] >= "20250302"
        ]

        # Process conference tournament games
        tournament_structures, conference_end_dates = load_tournament_structures(
            base_path, year
        )
        for game in tournament_games:
            # Skip if teams don't exist in conference_teams
            if (
                game["team1_id"] not in conference_teams
                or game["team2_id"] not in conference_teams
            ):
                continue

            team1_conf = conference_teams[game["team1_id"]].conference
            team2_conf = conference_teams[game["team2_id"]].conference

            if team1_conf == team2_conf:
                tournament = tournament_structures.get(team1_conf)
                if not tournament:
                    continue

                round_name = determine_tournament_round(tournament)

                conference_games.append(
                    ConferenceGameResult(
                        simulation_number=sim_number,
                        conference=team1_conf,
                        round=round_name,
                        team1_id=game["team1_id"],
                        team2_id=game["team2_id"],
                        team1_score=game["team1_score"],
                        team2_score=game["team2_score"],
                        winner_id=(
                            game["team1_id"]
                            if game["team1_score"] > game["team2_score"]
                            else game["team2_id"]
                        ),
                    )
                )

        # Get provisional teams
        provisional_teams = set()
        for structure in tournament_structures.values():
            provisional_teams.update(structure.provisional_teams)

        # Get auto bid recipients
        auto_bid_recipients = set()

        for team_id, team_stats in final_teams.items():
            if not team_stats["has_games"]:
                continue

            team_conf = conference_teams.get(team_id)
            if not team_conf:
                continue

            conf_name = team_conf.conference

            # Check if this team is a conference champion
            is_champion = (
                conf_name in conference_champions
                and conference_champions[conf_name] == team_id
            )

            # Award auto bid to conference champions
            if is_champion:
                auto_bid_recipients.add(team_id)

            # Special case for UAA
            elif conf_name == "UAA" and is_uaa_regular_season_champion(
                team_id, season_results, conference_teams
            ):
                auto_bid_recipients.add(team_id)

        # Analyze bid thieves
        bid_thieves, _ = analyze_tournament_bid_thieves(
            tournament_games,
            conference_champions,
            conference_teams,
            final_teams,
            auto_bid_recipients,
            provisional_teams,
        )

        # Now process all teams for complete results
        for team_id, team_stats in final_teams.items():
            if not team_stats["has_games"]:
                continue

            team_name = valid_teams.get(team_id, f"Unknown-{team_id}")
            wins = team_stats["wins"]
            losses = team_stats["losses"]
            npi = team_stats["npi"]

            got_auto_bid = team_id in auto_bid_recipients
            got_at_large = determine_at_large_bid(
                team_id, final_teams, auto_bid_recipients, provisional_teams
            )

            made_tournament = got_auto_bid or got_at_large

            # Check if team is in top N based on final NPI rankings
            made_top_16 = is_in_top_16(team_id, final_teams)
            made_top_8 = is_in_top_8(team_id, final_teams)
            made_top_4 = is_in_top_4(team_id, final_teams)

            results[team_id] = TeamSimResult(
                simulation_number=sim_number,
                team_name=team_name,
                wins=wins,
                losses=losses,
                npi_rank=npi,
                got_auto_bid=got_auto_bid,
                got_at_large=got_at_large,
                made_tournament=made_tournament,
                made_top_4=made_top_4,
                made_top_8=made_top_8,
                made_top_16=made_top_16,
            )

        # Get at-large bids for tournament analysis
        at_large_bids = {
            team_id for team_id, result in results.items() if result.got_at_large
        }

        tournament_results = get_conference_tournament_results(
            tournament_games, conference_teams, conference_champions, at_large_bids
        )

        return results, bid_thieves, tournament_results, conference_games

    except Exception as e:
        print(f"Error in simulation {sim_number}: {e}")
        print("Traceback:")
        import traceback

        traceback.print_exc()
        return {}, set(), {}, []


def calculate_team_stats(
    all_results: Dict[str, List[TeamSimResult]],
    tournament_stats: Dict[str, ConferenceTournamentStats],
) -> Dict[str, TeamStats]:
    """Calculate statistics across all simulations for each team."""
    team_stats = {}
    print(f"\nProcessing results for {len(all_results)} teams")

    # Debug counters for Millsaps
    millsaps_auto_bids = 0
    millsaps_total_sims = 0

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
        top_4_appearances = sum(1 for r in results if r.made_top_4)
        top_8_appearances = sum(1 for r in results if r.made_top_8)
        top_16_appearances = sum(1 for r in results if r.made_top_16)
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
            f_pool_c_pct=f_pool_c_pct,
            top_4_pct=(top_4_appearances / total_sims) * 100,
            top_8_pct=(top_8_appearances / total_sims) * 100,
            top_16_pct=(top_16_appearances / total_sims) * 100,
        )

    # Second pass - assign ranks based on tournament_pct
    sorted_teams = sorted(
        team_stats.items(), key=lambda x: x[1].tournament_pct, reverse=True
    )

    # Assign ranks (1-based ranking)
    for rank, (team_id, stats) in enumerate(sorted_teams, 1):
        team_stats[team_id] = replace(stats, rank=rank)

    print(f"\nCalculated stats for {len(team_stats)} teams")
    return team_stats


def run_multiple_simulations(base_path: Path, year: str, num_sims: int = 1000) -> Tuple[
    Dict[str, TeamStats],
    Dict[str, int],
    List[int],
    Dict[str, ConferenceTournamentStats],
]:
    all_results = defaultdict(list)
    bid_thief_counts = defaultdict(int)
    per_sim_bid_count = []
    tournament_stats = defaultdict(ConferenceTournamentStats)

    # Create CSV files for new data tracking
    conf_games_file = base_path / f"conference_games_{year}.csv"
    sim_details_file = base_path / f"simulation_details_{year}.csv"

    conf_writer = None
    sim_writer = None

    try:
        # Open files and create writers
        cgf = open(conf_games_file, "w", newline="")
        sdf = open(sim_details_file, "w", newline="")

        conf_writer = csv.writer(cgf)
        sim_writer = csv.writer(sdf)

        # Write headers
        conf_writer.writerow(
            [
                "simulation_number",
                "conference",
                "round",
                "team1_id",
                "team2_id",
                "team1_score",
                "team2_score",
                "winner_id",
            ]
        )
        sim_writer.writerow(
            [
                "simulation_number",
                "team_name",
                "wins",
                "losses",
                "npi_rank",
                "got_auto_bid",
                "got_at_large",
                "made_tournament",
                "made_top_4",
                "made_top_8",
                "made_top_16",
            ]
        )

        total_start_time = time.time()

        for sim_number in range(1, num_sims + 1):
            try:
                sim_start_time = time.time()
                print(f"\nStarting simulation {sim_number}")
                print(" Running season simulation...", flush=True)

                # Run simulation
                sim_results, sim_bid_thieves, tourn_results, conf_games = (
                    run_single_simulation(base_path, year, sim_number)
                )

                valid_teams = load_teams(base_path, year)

                # Write conference games to CSV
                for game in conf_games:
                    conf_writer.writerow(
                        [
                            game.simulation_number,
                            game.conference,
                            game.round,
                            valid_teams[game.team1_id],
                            valid_teams[game.team2_id],
                            game.team1_score,
                            game.team2_score,
                            game.winner_id,
                        ]
                    )

                # Write simulation results
                for team_id, result in sim_results.items():
                    sim_writer.writerow(
                        [
                            sim_number,
                            valid_teams[team_id],
                            result.wins,
                            result.losses,
                            result.npi_rank,
                            result.got_auto_bid,
                            result.got_at_large,
                            result.made_tournament,
                            result.made_top_4,
                            result.made_top_8,
                            result.made_top_16,
                        ]
                    )

                # Process tournament results
                for team_id, result in tourn_results.items():
                    stats = tournament_stats[team_id]
                    if result.exit_round == "Final":
                        stats.final_total += 1
                        if result.got_pool_c:
                            stats.final_pool_c += 1
                    elif result.exit_round == "Semifinal":
                        stats.semifinal_total += 1
                        if result.got_pool_c:
                            stats.semifinal_pool_c += 1
                    elif result.exit_round == "Quarterfinal":
                        stats.quarterfinal_total += 1
                        if result.got_pool_c:
                            stats.quarterfinal_pool_c += 1

                # Update other tracking
                for team_id, result in sim_results.items():
                    all_results[team_id].append(result)
                for team_id in sim_bid_thieves:
                    bid_thief_counts[team_id] += 1
                per_sim_bid_count.append(len(sim_bid_thieves))

                sim_duration = time.time() - sim_start_time
                print(
                    f"Completed simulation {sim_number} in {sim_duration:.2f} seconds"
                )

            except Exception as e:
                print(f"Error in simulation {sim_number}:")
                print(f" {str(e)}")
                traceback.print_exc()
                continue

        total_duration = time.time() - total_start_time
        print(f"\nAll simulations completed in {total_duration:.2f} seconds")
        print(f"Average time per simulation: {total_duration/num_sims:.2f} seconds")

        team_stats = calculate_team_stats(all_results, tournament_stats)
        return team_stats, bid_thief_counts, per_sim_bid_count, tournament_stats

    finally:
        # Clean up - close files
        if conf_writer is not None:
            cgf.close()
        if sim_writer is not None:
            sdf.close()


def save_simulation_stats(
    stats: Dict[str, TeamStats],
    base_path: Path,
    year: str,
    conference_teams: Dict[str, ConferenceTeam],
):
    """Save simulation statistics to CSV with team names and conferences."""
    output_path = base_path / year / "simulation_stats.csv"
    teams_mapping = {}
    mapping_path = base_path / year / "teams_mapping.txt"
    try:
        with open(mapping_path, "r") as file:
            next(file)
            for line in file:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    team_id = parts[0].strip()
                    scott_name = parts[2].strip()
                    teams_mapping[team_id] = scott_name
    except Exception as e:
        print(f"Error loading team mappings: {e}")
        return

    # Sort by median rank
    sorted_teams = sorted(stats.items(), key=lambda x: x[1].median_rank, reverse=True)

    # Debug top teams before writing
    print("\nTop 5 teams by NPI:")
    for team_id, team_stats in sorted_teams[:5]:
        team_name = teams_mapping.get(team_id, f"Unknown ({team_id})")
        conference = conference_teams[team_id].conference
        print(f"{team_name} ({conference}): NPI = {team_stats.median_rank:.2f}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Team",
                "Conf",
                "MedW",
                "MedL",
                "Tourn%",
                "A%",
                "C%",
                "ALWYNI%",
                "F-C%",
                "SF-C%",
                "QF-C%",
                "Top4%",
                "Top8%",
                "Top16%",
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

            def format_pool_c_pct(pct):
                if isinstance(pct, str):
                    return pct
                return f"{pct:.1f}%"

            writer.writerow(
                [
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
                    f"{team_stats.top_4_pct:.1f}%",
                    f"{team_stats.top_8_pct:.1f}%",
                    f"{team_stats.top_16_pct:.1f}%",
                    f"{team_stats.median_rank:.1f}",
                    f"{team_stats.min_rank:.1f}",
                    f"{team_stats.max_rank:.1f}",
                    team_stats.rank,
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

    # Load tournament structures to get the end dates for each conference
    base_path = Path(__file__).parent / "data"
    year = "2025"  # This should be dynamically determined based on your code
    tournament_structures, conference_end_dates = load_tournament_structures(
        base_path, year
    )

    # Group tournament games by conference using conference-specific end dates
    conference_games = defaultdict(list)
    for game in season_results:
        # Skip games with missing team IDs
        if (
            game["team1_id"] not in conference_teams
            or game["team2_id"] not in conference_teams
        ):
            continue

        team1_conf = conference_teams[game["team1_id"]].conference
        team2_conf = conference_teams[game["team2_id"]].conference

        # Skip UAA conference games (as you noted, this is correct)
        if team1_conf == "UAA" or team2_conf == "UAA":
            continue

        # Only consider games within the same conference
        if team1_conf == team2_conf:
            conf = team1_conf

            # Check if this game is after the conference's regular season end date
            if conf in conference_end_dates:
                end_date = conference_end_dates[conf]
                if game["date"] > end_date:
                    conference_games[conf].append(game)

    # Debug output to verify
    print(f"Conferences with tournament games: {len(conference_games)}")
    for conf, games in conference_games.items():
        print(f"  {conf}: {len(games)} games")

    # Check SAA specifically
    if "SAA" in conference_games:
        print(f"SAA games found: {len(conference_games['SAA'])}")
        for game in sorted(conference_games["SAA"], key=lambda x: x["date"]):
            print(
                f"  {game['date']}: {game['team1_id']} vs {game['team2_id']} ({game['team1_score']}-{game['team2_score']})"
            )

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

            # Store as conference -> winner_id
            champions[conf] = winner_id

    return champions


def get_conference_tournament_results(
    tournament_games: List[dict],
    conference_teams: Dict[str, ConferenceTeam],
    conference_champions: Set[str],
    at_large_bids: Set[str],
) -> Dict[str, TeamTournamentResult]:
    """Analyze conference tournament performance and Pool C bid correlation."""
    results: Dict[str, TeamTournamentResult] = {}

    # First group games by conference
    conference_games: Dict[str, List[dict]] = {}
    for game in tournament_games:
        team1_id = game["team1_id"]
        team2_id = game["team2_id"]
        if team1_id not in conference_teams or team2_id not in conference_teams:
            continue

        conf = conference_teams[team1_id].conference
        if conf not in conference_games:
            conference_games[conf] = []
        conference_games[conf].append(game)

    # Process each conference tournament
    for conf, games in conference_games.items():

        # Sort games by date and count games per date
        games_by_date = {}
        for game in sorted(games, key=lambda x: x["date"]):
            date = game["date"]
            if date not in games_by_date:
                games_by_date[date] = []
            games_by_date[date].append(game)

        dates = sorted(games_by_date.keys())
        games_per_date = {date: len(games_by_date[date]) for date in dates}

        # Map each date to a round
        round_by_date = {}
        remaining_dates = sorted(dates)

        # Find quarterfinal date
        qf_games = [d for d in dates if games_per_date[d] >= 4]
        if qf_games:
            qf_date = min(qf_games)
            round_by_date[qf_date] = "Quarterfinal"
            remaining_dates.remove(qf_date)

        # Find final date
        final_candidates = [d for d in remaining_dates if games_per_date[d] == 1]
        if final_candidates:
            final_date = max(final_candidates)
            round_by_date[final_date] = "Final"
            if final_date in remaining_dates:
                remaining_dates.remove(final_date)

        # Assign semifinals
        for date in remaining_dates:
            round_by_date[date] = "Semifinal"

        # Process each game
        eliminated_teams = set()
        for date in dates:
            if date not in round_by_date:
                print(f"WARNING: Date {date} has no round assignment")
                continue

            round_name = round_by_date[date]
            for game in games_by_date[date]:
                team1_id = game["team1_id"]
                team2_id = game["team2_id"]
                team1_score = game["team1_score"]
                team2_score = game["team2_score"]

                team1_is_winner = team1_score > team2_score
                winner_id = team1_id if team1_is_winner else team2_id
                loser_id = team2_id if team1_is_winner else team1_id

                if loser_id not in eliminated_teams:
                    eliminated_teams.add(loser_id)
                    results[loser_id] = TeamTournamentResult(
                        team_id=loser_id,
                        conference=conf,
                        exit_round=round_name,
                        got_pool_c=loser_id in at_large_bids,
                    )

                if round_name == "Final":
                    # Remove recording of winner - only record the loser
                    if loser_id not in eliminated_teams:
                        eliminated_teams.add(loser_id)
                        results[loser_id] = TeamTournamentResult(
                            team_id=loser_id,
                            conference=conf,
                            exit_round=round_name,
                            got_pool_c=loser_id in at_large_bids,
                        )

    return results


def determine_tournament_round(tournament: ConferenceTournament) -> str:
    """
    Determine the round of a conference tournament game based on the structure.

    Args:
        tournament: ConferenceTournament object for the conference

    Returns:
        str: "First Round", "Quarterfinal", "Semifinal", or "Final"
    """
    # Calculate number of rounds needed based on total teams and byes
    teams_playing = tournament.total_teams - tournament.byes

    if teams_playing <= 2:
        return "Final"
    elif teams_playing <= 4:
        return "Semifinal"
    elif teams_playing <= 8:
        return "Quarterfinal"
    else:
        return "First Round"


def is_in_top_n(team_id: str, final_teams: Dict[str, Dict], n: int) -> bool:
    """
    Determine if a team is in the top N based on their final NPI rank.

    Args:
        team_id: Team ID to check
        final_teams: Dictionary of team stats including NPI rankings
        n: Number of top teams to check (e.g., 4, 8, 16)

    Returns:
        bool: True if team is in top N by NPI rank
    """
    # First, get all teams that have games
    valid_teams = {
        tid: stats
        for tid, stats in final_teams.items()
        if stats.get("has_games", False)
    }

    # Sort teams by NPI
    sorted_teams = sorted(
        valid_teams.items(),
        key=lambda x: x[1]["npi"],
        reverse=True,  # Higher NPI is better
    )

    # Get top N team IDs
    top_n_teams = {team_id for team_id, _ in sorted_teams[:n]}

    return team_id in top_n_teams


def is_in_top_4(team_id: str, final_teams: Dict[str, Dict]) -> bool:
    """Determine if a team finished in the top 4 by NPI rank."""
    return is_in_top_n(team_id, final_teams, 4)


def is_in_top_8(team_id: str, final_teams: Dict[str, Dict]) -> bool:
    """Determine if a team finished in the top 8 by NPI rank."""
    return is_in_top_n(team_id, final_teams, 8)


def is_in_top_16(team_id: str, final_teams: Dict[str, Dict]) -> bool:
    """Determine if a team finished in the top 16 by NPI rank."""
    return is_in_top_n(team_id, final_teams, 16)
