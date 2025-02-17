import csv
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans


@dataclass
class NCAABracketTeam:
    team: str
    overall_seed: int
    conference: str
    region: int


@dataclass
class TeamDistance:
    team_a: str
    team_b: str
    distance: float


@dataclass
class NCAABracketMatchup:
    team_a: str
    team_a_actual_seed: int
    team_b: str
    team_b_actual_seed: int
    pod_name: str
    quadrant_name: str
    team_a_true_seed: int
    team_a_seed_diff: float
    team_b_true_seed: int
    team_b_seed_diff: float
    team_a_conf: str
    team_b_conf: str
    team_a_reg: int
    team_b_reg: int


class DataLoader:
    @staticmethod
    def load_teams(file_path: Path) -> List[NCAABracketTeam]:
        """
        Loads team data from the specified file.
        Expected format: Team,Overall_seed,Conf,Region
        """
        teams = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                teams.append(
                    NCAABracketTeam(
                        team=row["Team"].strip(),
                        overall_seed=int(row["Overall_seed"]),
                        conference=row["Conf"].strip(),
                        region=int(row["Region"]),
                    )
                )
        return teams

    @staticmethod
    def load_distances(file_path: Path) -> List[TeamDistance]:
        """
        Loads distance data from the specified file.
        Expected format: Origin_School,Destination_School,Distance
        """
        distances = []
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                distances.append(
                    TeamDistance(
                        team_a=row["Origin_School"].strip(),
                        team_b=row["Destination_School"].strip(),
                        distance=float(row["Distance"]),
                    )
                )
        return distances


class BracketGenerator:
    def __init__(
        self,
        base_path: Path,
        year: str,
        regular_season_matchups: List[Tuple[str, str]] = None,
    ):
        self.year = year
        self.regular_season_matchups = set(regular_season_matchups or [])
        self.quadrants = ["Q1", "Q2", "Q3", "Q4"]

        # Load data
        year_path = base_path / year
        teams = DataLoader.load_teams(year_path / "ncaa_bracket_teams.txt")
        distances = DataLoader.load_distances(year_path / "team_distances.txt")

        # Initialize data structures
        self.teams = teams
        self._distances = self._create_distance_map(distances)
        self._team_lookup = {team.team: team for team in teams}
        self._validate_data()

    def _create_distance_map(
        self, distances: List[TeamDistance]
    ) -> Dict[Tuple[str, str], float]:
        """Creates a bidirectional map of team distances."""
        distance_map = {}
        for d in distances:
            distance_map[(d.team_a, d.team_b)] = d.distance
            distance_map[(d.team_b, d.team_a)] = d.distance
        return distance_map

    def _validate_data(self):
        """Validates the loaded data for consistency."""
        # Check for duplicate teams
        if len(self.teams) != len(set(team.team for team in self.teams)):
            raise ValueError("Duplicate team names found in team data")

        # Check for valid overall seeds (1-64)
        seeds = [team.overall_seed for team in self.teams]
        if not all(1 <= seed <= 64 for seed in seeds):
            raise ValueError(
                "Invalid overall seed found. Seeds must be between 1 and 64"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate overall seeds found")

        # Check for valid regions (appears to be 1-10 based on sample)
        if not all(1 <= team.region <= 10 for team in self.teams):
            raise ValueError("Invalid region found. Regions must be between 1 and 10")

    def get_team(self, team_name: str) -> Optional[NCAABracketTeam]:
        """Gets team data by team name."""
        return self._team_lookup.get(team_name)

    def get_distance(self, team_a: str, team_b: str) -> float:
        """Gets distance between two teams."""
        return self._distances.get((team_a, team_b), float("inf"))

    def get_teams_by_seed_range(
        self, start_seed: int, end_seed: int
    ) -> List[NCAABracketTeam]:
        """Gets all teams within the specified seed range, inclusive."""
        return sorted(
            [
                team
                for team in self.teams
                if start_seed <= team.overall_seed <= end_seed
            ],
            key=lambda x: x.overall_seed,
        )

    def score_matchup(
        self,
        team_a: NCAABracketTeam,
        team_b: NCAABracketTeam,
        actual_seed_a: int,
        actual_seed_b: int,
    ) -> float:
        """
        Scores a single matchup based on multiple criteria.
        Lower score is better.
        """
        score = 0.0

        # Seeding penalty
        seed_diff_a = abs(self.get_quadrant_seed(team_a.overall_seed) - actual_seed_a)
        seed_diff_b = abs(self.get_quadrant_seed(team_b.overall_seed) - actual_seed_b)
        score += (seed_diff_a + seed_diff_b) * 0.75

        # Non-conference rematch penalty
        if (team_a.team, team_b.team) in self.regular_season_matchups or (
            team_b.team,
            team_a.team,
        ) in self.regular_season_matchups:
            score += 3.0

        # Flight penalty
        distance = self.get_distance(team_a.team, team_b.team)
        if distance > 500:
            score += 750.0

        # Same region penalty
        if team_a.region == team_b.region:
            score += 1.0

        # Same conference penalty
        if team_a.conference == team_b.conference:
            score += 1220

        return score

    def get_quadrant_seed(self, overall_seed: int) -> int:
        """Calculate quadrant seed (1-16) from overall seed (1-64)"""
        return ((overall_seed - 1) // 4) + 1

    def _assign_top_seeds_to_quadrants(self) -> Dict[str, List[NCAABracketTeam]]:
        """
        Assigns teams to quadrants following tournament rules.
        Each group of 4 teams by overall seed (1-4, 5-8, etc.) should be split across the quadrants.
        """
        # First, validate we have exactly 64 teams
        if len(self.teams) != 64:
            raise ValueError(f"Expected 64 teams, but got {len(self.teams)}")

        quadrant_assignments = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

        # Track all teams we use to ensure none are missed
        used_teams = set()

        # Sort teams by overall seed
        sorted_teams = sorted(self.teams, key=lambda x: x.overall_seed)

        # Find the actual top 4 overall seeds (lowest overall seeds)
        top_4_seeds = sorted_teams[:4]

        # First group of 4 has special assignment:
        initial_assignments = [
            ("Q1", top_4_seeds[0]),  # 1st lowest overall seed
            ("Q3", top_4_seeds[1]),  # 2nd lowest overall seed
            ("Q4", top_4_seeds[2]),  # 3rd lowest overall seed
            ("Q2", top_4_seeds[3]),  # 4th lowest overall seed
        ]

        for quadrant, team in initial_assignments:
            quadrant_assignments[quadrant].append(team)
            used_teams.add(team.team)

        # For remaining groups of 4, assign in order to Q1, Q2, Q3, Q4
        for i in range(4, len(sorted_teams), 4):
            group = sorted_teams[i : i + 4]
            for team, quadrant in zip(group, ["Q1", "Q2", "Q3", "Q4"]):
                quadrant_assignments[quadrant].append(team)
                used_teams.add(team.team)

        # Verify we used all teams
        all_team_names = {team.team for team in self.teams}
        missed_teams = all_team_names - used_teams
        if missed_teams:
            raise ValueError(f"Failed to assign these teams: {missed_teams}")

        # Verify each quadrant has 16 teams
        for quadrant, teams in quadrant_assignments.items():
            if len(teams) != 16:
                raise ValueError(f"{quadrant} has {len(teams)} teams instead of 16")

        return quadrant_assignments

    def _create_pod_matchups(
        self, quadrant: str, teams: List[NCAABracketTeam]
    ) -> List[NCAABracketMatchup]:
        """
        Create matchups with predefined seed assignments
        Seed pairings for each pod:
        P1: 1 vs 16, 8 vs 9
        P2: 5 vs 12, 4 vs 13
        P3: 6 vs 11, 3 vs 14
        P4: 7 vs 10, 2 vs 15
        """
        # Predefined seed pairings
        pod_seed_pairings = {
            "P1": [(1, 16), (8, 9)],
            "P2": [(5, 12), (4, 13)],
            "P3": [(6, 11), (3, 14)],
            "P4": [(7, 10), (2, 15)],
        }

        # Sort teams by overall seed
        teams_by_seed = sorted(teams, key=lambda x: x.overall_seed)

        matchups = []
        used_teams = set()

        for pod_num in ["P1", "P2", "P3", "P4"]:
            pod_name = f"{quadrant}_{pod_num}"
            seed_pairs = pod_seed_pairings[pod_num]

            for high_seed, low_seed in seed_pairs:
                # Find teams that best match the seed pairing based on quadrant seed
                high_team = min(
                    (team for team in teams_by_seed if team.team not in used_teams),
                    key=lambda x: abs(
                        self.get_quadrant_seed(x.overall_seed) - high_seed
                    ),
                )

                low_team = min(
                    (team for team in teams_by_seed if team.team not in used_teams),
                    key=lambda x: abs(
                        self.get_quadrant_seed(x.overall_seed) - low_seed
                    ),
                )

                # Mark teams as used
                used_teams.update([high_team.team, low_team.team])

                # Create matchup
                matchups.append(
                    NCAABracketMatchup(
                        team_a=high_team.team,
                        team_a_actual_seed=self.get_quadrant_seed(
                            high_team.overall_seed
                        ),  # Use quadrant seed for actual seed
                        team_b=low_team.team,
                        team_b_actual_seed=self.get_quadrant_seed(
                            low_team.overall_seed
                        ),  # Use quadrant seed for actual seed
                        pod_name=pod_name,
                        quadrant_name=quadrant,
                        team_a_true_seed=self.get_quadrant_seed(high_team.overall_seed),
                        team_a_seed_diff=abs(
                            self.get_quadrant_seed(high_team.overall_seed) - high_seed
                        ),
                        team_b_true_seed=self.get_quadrant_seed(low_team.overall_seed),
                        team_b_seed_diff=abs(
                            self.get_quadrant_seed(low_team.overall_seed) - low_seed
                        ),
                        team_a_conf=high_team.conference,
                        team_b_conf=low_team.conference,
                        team_a_reg=high_team.region,
                        team_b_reg=low_team.region,
                    )
                )

        # Validate all teams used
        if len(used_teams) != len(teams):
            unused_teams = {team.team for team in teams} - used_teams
            raise ValueError(
                f"Not all teams used in {quadrant}! Unused teams: {unused_teams}"
            )

        return matchups

    def create_initial_bracket(self) -> List[NCAABracketMatchup]:
        """Creates the initial bracket."""
        quadrant_assignments = self._assign_top_seeds_to_quadrants()

        all_matchups = []
        for quadrant, teams in quadrant_assignments.items():
            quadrant_matchups = self._create_pod_matchups(quadrant, teams)
            all_matchups.extend(quadrant_matchups)

        # 🔍 Debug: Check the matchups right before returning
        print("\nFinal matchups before returning from create_initial_bracket():")
        for matchup in all_matchups:
            print(
                f"{matchup.team_a} ({matchup.team_a_actual_seed}) vs {matchup.team_b} ({matchup.team_b_actual_seed}) | {matchup.pod_name}"
            )

        return all_matchups

    def score_bracket(self, matchups: List[NCAABracketMatchup]) -> float:
        """
        Scores entire bracket based on multiple criteria.
        Lower score is better (less penalty).
        """
        total_score = 0.0

        for matchup in matchups:
            team_a = self.get_team(matchup.team_a)
            team_b = self.get_team(matchup.team_b)
            total_score += self.score_matchup(
                team_a, team_b, matchup.team_a_actual_seed, matchup.team_b_actual_seed
            )

        return total_score

    def assign_regional_pods(self, teams: List[NCAABracketTeam], num_regions: int = 8):
        """
        Groups teams into regional pods based on geographic proximity.
        Ensures that teams are initially placed in geographically logical regions.
        """

        # Convert team locations into coordinates
        team_locations = np.array([(team.latitude, team.longitude) for team in teams])

        # Cluster teams into `num_regions` regions using K-Means
        kmeans = KMeans(n_clusters=num_regions, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(team_locations)

        # Assign each team to a regional pod
        for i, team in enumerate(teams):
            team.region = clusters[i]

        return teams

    def assign_dynamic_regions(self, teams: List[NCAABracketTeam]):
        """
        Dynamically assigns teams to regional pods based on actual travel distances.
        Uses DBSCAN to form natural clusters instead of forcing an equal number of regions.
        """
        coords = np.array([(team.latitude, team.longitude) for team in teams])

        # DBSCAN parameters: eps controls cluster tightness, min_samples ensures enough teams per region
        clustering = DBSCAN(eps=500, min_samples=4, metric="euclidean").fit(coords)

        for i, team in enumerate(teams):
            team.region = clustering.labels_[i]  # Assign each team to a region

        return teams

    def optimize_bracket(
        self, initial_bracket: List[NCAABracketMatchup], iterations: int = 1000
    ) -> List[NCAABracketMatchup]:
        """
        Optimizes the bracket by iteratively swapping teams to improve the overall score.
        Uses simulated annealing, prioritizing pod-based travel and region balance.
        """

        current_bracket = initial_bracket.copy()
        current_score = self.score_bracket(current_bracket)
        best_bracket = current_bracket
        best_score = current_score

        # Simulated annealing parameters
        initial_temperature = 50.0
        cooling_rate = 0.95
        temperature = initial_temperature

        max_flights_allowed = 4  # Hard limit on flights

        # Step 1: Group matchups into 4-team pods
        pods = [
            current_bracket[i : i + 2] for i in range(0, len(current_bracket), 2)
        ]  # Each pod contains 2 matchups

        for iteration in range(iterations):
            modified_bracket = current_bracket.copy()
            modified_pods = [
                modified_bracket[i : i + 2] for i in range(0, len(modified_bracket), 2)
            ]  # Regenerate pods

            try:
                # Select a pod to modify
                pod1 = random.choice(modified_pods)

                # Find another pod **preferably within 500 miles**
                nearby_pods = [
                    pod
                    for pod in modified_pods
                    if self.get_distance(pod1[0].team_a, pod[0].team_a) < 500
                ]

                # If we found a nearby pod, prioritize swapping within it
                if nearby_pods:
                    pod2 = random.choice(nearby_pods)
                else:
                    pod2 = random.choice(modified_pods)  # Fallback to random pod swap

                # Swap only one team between pods
                if random.random() < 0.5:
                    pod1[0].team_a, pod2[0].team_a = pod2[0].team_a, pod1[0].team_a
                else:
                    pod1[1].team_b, pod2[1].team_b = pod2[1].team_b, pod1[1].team_b

                # Step 2: Recalculate flights at the pod level
                current_flights = sum(
                    self.calculate_flights_per_pod(pod) for pod in pods
                )
                new_flights = sum(
                    self.calculate_flights_per_pod(pod) for pod in modified_pods
                )

                # **Enforce a hard cap on flights**
                if new_flights > max_flights_allowed:
                    continue  # Reject this swap if flights exceed limit

                # **Heavy penalty for flight increase**
                flight_penalty_weight = 1000
                flight_score_delta = (
                    new_flights - current_flights
                ) * flight_penalty_weight
                new_score = self.score_bracket(modified_bracket) + flight_score_delta

                # **Region balance score**: favor brackets where regions stay balanced
                region_balance_score = sum(
                    abs(
                        len(
                            [
                                match
                                for match in modified_bracket
                                if match.team_a.region == r
                            ]
                        )
                        - len(
                            [
                                match
                                for match in modified_bracket
                                if match.team_b.region == r
                            ]
                        )
                    )
                    for r in set(match.team_a.region for match in modified_bracket)
                )

                # Add region balance penalty
                new_score += region_balance_score * 10  # Adjust weight as needed

                # **Reduce Seeding Penalty Further**
                seeding_weight = 0.2  # Allows for even more flexibility

                # Calculate acceptance probability (simulated annealing)
                delta_score = new_score - current_score
                acceptance_probability = (
                    math.exp(-delta_score / temperature) if delta_score > 0 else 1.0
                )

                # Accept swap if score improves or with some probability
                if random.random() < acceptance_probability:
                    current_bracket = modified_bracket
                    current_score = new_score
                    pods = modified_pods  # Update pods

                    # Update best bracket if needed
                    if current_score < best_score:
                        best_bracket = current_bracket
                        best_score = current_score

            except Exception:
                continue  # Skip failed swap attempts

            # Cool down temperature
            temperature *= cooling_rate

        return best_bracket

    def calculate_flights_per_pod(self, pod: List[NCAABracketMatchup]) -> int:
        """
        Determines the number of flights required for a given 4-team pod.
        A flight is counted if any team must fly to the pod host.
        """
        # The host should ideally be the highest-seeded team (1 or 2)
        host_team = min(
            [match.team_a for match in pod] + [match.team_b for match in pod],
            key=lambda team: team.overall_seed,
        )

        flights = sum(
            1
            for match in pod
            for team in [match.team_a, match.team_b]
            if self.get_distance(team, host_team) > 500
        )

        return flights

    def generate_bracket(self) -> List[NCAABracketMatchup]:
        """Main function to generate and optimize bracket."""
        initial_bracket = self.create_initial_bracket()
        optimized_bracket = self.optimize_bracket(initial_bracket)
        return optimized_bracket


def write_bracket_to_file(bracket: List[NCAABracketMatchup], filename: str):
    """Writes bracket to specified file in requested format."""
    with open(filename, "w") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "team_a",
                "team_a_actual_seed",
                "team_b",
                "team_b_actual_seed",
                "pod_name",
                "quadrant_name",
                "team_a_true_seed",
                "team_a_seed_diff",
                "team_b_true_seed",
                "team_b_seed_diff",
                "team_a_conf",
                "team_b_conf",
                "team_a_reg",
                "team_b_reg",
            ]
        )

        # Write matchups directly
        for matchup in bracket:
            writer.writerow(
                [
                    matchup.team_a,
                    matchup.team_a_actual_seed,
                    matchup.team_b,
                    matchup.team_b_actual_seed,
                    matchup.pod_name,
                    matchup.quadrant_name,
                    matchup.team_a_true_seed,
                    matchup.team_a_seed_diff,
                    matchup.team_b_true_seed,
                    matchup.team_b_seed_diff,
                    matchup.team_a_conf,
                    matchup.team_b_conf,
                    matchup.team_a_reg,
                    matchup.team_b_reg,
                ]
            )
