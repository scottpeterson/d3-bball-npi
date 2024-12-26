from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import random


@dataclass
class GameResult:
    winner_id: str
    loser_id: str
    winning_score: int
    losing_score: int
    was_upset: bool = False


class EloSimulator:
    def __init__(self, k_factor: float = 32.0, home_advantage: float = 100):
        """
        Initialize the ELO simulator

        Args:
            k_factor: How quickly ratings change (default: 32.0)
            home_advantage: ELO points added for home court (default: 100)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.team_ratings: Dict[str, float] = {}

    def efficiency_to_elo(self, efficiency: float) -> float:
        """
        Convert efficiency margin to ELO rating.
        Using a scaling factor where each 1 point of efficiency ≈ 25 ELO points
        Base rating of 1500
        """
        return 1500 + (efficiency * 25)

    def initialize_ratings(self, team_data: Dict[str, Tuple[float, float]]):
        """
        Initialize ELO ratings from efficiency margins

        Args:
            team_data: Dictionary mapping team_id to tuple of (adjEM, adjT)
        """
        for team_id, (adj_em, _) in team_data.items():
            self.team_ratings[team_id] = self.efficiency_to_elo(adj_em)

    def probability(self, rating1: float, rating2: float) -> float:
        """
        Calculate expected score (win probability) for rating1 against rating2
        """
        return 1.0 / (1.0 + math.pow(10, (rating2 - rating1) / 400.0))

    def calculate_win_probability(
        self, team_a_id: str, team_b_id: str, team_b_home: bool = True
    ) -> Dict[str, float]:
        """
        Calculate win probability using ELO ratings

        Args:
            team_a_id: ID of first team
            team_b_id: ID of second team
            team_b_home: Whether team B is home (default: True)

        Returns:
            Dictionary mapping team IDs to win probabilities
        """
        rating_a = self.team_ratings[team_a_id]
        rating_b = self.team_ratings[team_b_id]

        # Apply home court advantage
        if team_b_home:
            rating_b += self.home_advantage

        # Calculate probabilities using correct formula
        prob_a = self.probability(rating_a, rating_b)
        prob_b = 1.0 - prob_a

        return {team_a_id: prob_a, team_b_id: prob_b}

    def update_ratings(self, winner_id: str, loser_id: str, team_b_home: bool = True):
        """
        Update ELO ratings after a game result using correct ELO formula

        Args:
            winner_id: ID of winning team
            loser_id: ID of losing team
            team_b_home: Whether team B was home
        """
        rating_winner = self.team_ratings[winner_id]
        rating_loser = self.team_ratings[loser_id]

        # Apply home court for probability calculation
        if team_b_home and winner_id == loser_id:
            rating_loser += self.home_advantage
        elif team_b_home:
            rating_winner += self.home_advantage

        # Calculate probabilities
        prob_winner = self.probability(rating_winner, rating_loser)
        prob_loser = self.probability(rating_loser, rating_winner)

        # Update ratings using correct formula
        # Winner won (outcome = 1), loser lost (outcome = 0)
        self.team_ratings[winner_id] = rating_winner + self.k_factor * (1 - prob_winner)
        self.team_ratings[loser_id] = rating_loser + self.k_factor * (0 - prob_loser)

    def simulate_game(
        self, team_a_id: str, team_b_id: str, team_b_home: bool = True
    ) -> GameResult:
        """
        Simulate a game between two teams

        Args:
            team_a_id: ID of first team
            team_b_id: ID of second team
            team_b_home: Whether team B is home

        Returns:
            GameResult object with winner and scores
        """
        probabilities = self.calculate_win_probability(
            team_a_id, team_b_id, team_b_home
        )

        # Generate random outcome
        random_value = random.random()
        team_a_wins = random_value < probabilities[team_a_id]

        # Generate plausible scores
        # Higher rated team tends to score more
        rating_diff = abs(self.team_ratings[team_a_id] - self.team_ratings[team_b_id])
        base_score = 65  # Average D3 score
        score_variance = 10  # Points of variance

        if team_a_wins:
            winner_id = team_a_id
            loser_id = team_b_id
            winner_rating = self.team_ratings[team_a_id]
            loser_rating = self.team_ratings[team_b_id]
            was_upset = probabilities[team_b_id] > 0.5
        else:
            winner_id = team_b_id
            loser_id = team_a_id
            winner_rating = self.team_ratings[team_b_id]
            loser_rating = self.team_ratings[team_a_id]
            was_upset = probabilities[team_a_id] > 0.5

        # Generate scores based on rating difference
        rating_factor = min(rating_diff / 400, 1.0)  # Cap the effect
        margin = random.gauss(8 * rating_factor, 4)
        winner_score = int(random.gauss(base_score + margin / 2, score_variance))
        loser_score = int(random.gauss(base_score - margin / 2, score_variance))

        # Ensure winner actually wins
        if winner_score <= loser_score:
            winner_score = loser_score + 1

        # Update ratings based on result using correct formula
        self.update_ratings(winner_id, loser_id, team_b_home)

        return GameResult(
            winner_id=winner_id,
            loser_id=loser_id,
            winning_score=winner_score,
            losing_score=loser_score,
            was_upset=was_upset,
        )
