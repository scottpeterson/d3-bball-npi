from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .conf_tournaments import ConferenceTeam


def analyze_tournament_bid_thieves(
    tournament_games: List[dict],
    conference_champions: Dict[str, str],
    conference_teams: Dict[str, ConferenceTeam],
    final_teams: Dict[str, dict],
    auto_bid_recipients: Set[str],
    provisional_teams: Set[str],
) -> Tuple[List[str], Dict[str, int]]:
    bid_thieves = []
    bid_thief_counts = {}

    for conf, champion_id in conference_champions.items():
        # Find championship game
        championship_game = next(
            (
                game
                for game in reversed(tournament_games)
                if (game["team1_id"] == champion_id or game["team2_id"] == champion_id)
                and (
                    conference_teams[game["team1_id"]].conference == conf
                    and conference_teams[game["team2_id"]].conference == conf
                )
            ),
            None,
        )

        if not championship_game:
            continue

        # Identify runner-up
        runner_up = (
            championship_game["team2_id"]
            if championship_game["team1_id"] == champion_id
            else championship_game["team1_id"]
        )

        # Check if runner-up would get Pool C
        would_runner_up_get_pool_c = determine_at_large_bid(
            runner_up,
            final_teams,
            auto_bid_recipients,
            provisional_teams,
        )

        if not would_runner_up_get_pool_c:
            continue

        # Create hypothetical scenario
        modified_auto_bids = (auto_bid_recipients - {champion_id}) | {runner_up}

        # Check if champion would get Pool C in this scenario
        would_get_pool_c = determine_at_large_bid(
            champion_id,
            final_teams,
            modified_auto_bids,
            provisional_teams,
        )

        if not would_get_pool_c:
            bid_thieves.append(champion_id)
            bid_thief_counts[champion_id] = bid_thief_counts.get(champion_id, 0) + 1

    return bid_thieves, bid_thief_counts


def determine_at_large_bid(
    team_id: str,
    final_teams: Dict[str, dict],
    auto_bid_recipients: Set[str],
    provisional_teams: Set[str],
) -> bool:
    if team_id in auto_bid_recipients or team_id in provisional_teams:
        return False

    ranked_teams = sorted(
        [
            (t_id, stats)
            for t_id, stats in final_teams.items()
            if t_id not in auto_bid_recipients
            and t_id not in provisional_teams
            and stats["has_games"]
        ],
        key=lambda x: x[1]["npi"],
        reverse=True,
    )

    remaining_spots = 64 - len(auto_bid_recipients)
    team_rank = next(
        (idx for idx, (t_id, _) in enumerate(ranked_teams) if t_id == team_id), None
    )
    return team_rank is not None and team_rank < remaining_spots
