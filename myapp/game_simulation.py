from dataclasses import dataclass
from typing import Dict, Tuple
import random


@dataclass
class GameResult:
    winner_id: str
    loser_id: str
    winning_score: int
    losing_score: int
    was_upset: bool


def simulate_game(
    team_data: Dict[str, Tuple[float, float]],
    team_a_id: str,
    team_b_id: str,
    home_advantage: float = 3.5,
) -> GameResult:
    """Core game simulation logic"""
    # [Your existing simulate_game implementation]
    # Calculate expected point differential
    team_a_stats = team_data[team_a_id]
    team_b_stats = team_data[team_b_id]
    adj_em_diff = team_a_stats[0] - team_b_stats[0]
    avg_tempo = (team_a_stats[1] + team_b_stats[1]) / 2
    expected_diff = adj_em_diff * (avg_tempo / 100) - home_advantage

    # Generate random point differential using normal distribution
    actual_diff = random.gauss(expected_diff, 11)

    # Calculate expected points per possession for each team
    base_efficiency = 102
    team_a_efficiency = base_efficiency + (team_a_stats[0] / 2)
    team_b_efficiency = base_efficiency - (team_a_stats[0] / 2)

    # Calculate expected points based on tempo and efficiency
    possessions = avg_tempo
    team_a_expected = (team_a_efficiency * possessions) / 100
    team_b_expected = (team_b_efficiency * possessions) / 100
    expected_total = team_a_expected + team_b_expected

    # Add random variation to total points
    total_points = random.gauss(expected_total, 10)

    if actual_diff > 0:
        winner_id = team_a_id
        loser_id = team_b_id
    else:
        winner_id = team_b_id
        loser_id = team_a_id
        actual_diff = -actual_diff

    # Calculate individual team scores
    winner_score = round((total_points + actual_diff) / 2)
    loser_score = round((total_points - actual_diff) / 2)

    # Ensure scores are reasonable
    if winner_score < 55:
        winner_score = round(random.uniform(55, 65))
    if loser_score < 55:
        loser_score = round(random.uniform(55, winner_score - 3))

    # Cap extremely high scores
    if winner_score > 100:
        winner_score = round(random.uniform(90, 100))
        loser_score = winner_score - round(actual_diff)
    if loser_score > 95:
        loser_score = round(random.uniform(85, 95))

    # Determine if it was an upset based on efficiency margins
    was_upset = (winner_id == team_b_id and adj_em_diff > 0) or (
        winner_id == team_a_id and adj_em_diff < 0
    )

    return GameResult(
        winner_id=winner_id,
        loser_id=loser_id,
        winning_score=winner_score,
        losing_score=loser_score,
        was_upset=was_upset,
    )
