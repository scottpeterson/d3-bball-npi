import csv
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

    def calculate_seed_difference(
        self, team: NCAABracketTeam, assigned_seed: int
    ) -> float:
        """
        Calculates the seed difference penalty for a team.
        Should be zero if the true quadrant seed matches the actual seed.
        """
        # Get the quadrant seed for this team
        team_quadrant_seed = self.get_quadrant_seed(team.overall_seed)

        # Calculate difference only if quadrant seeds don't match
        return -abs(team_quadrant_seed - assigned_seed) * 0.75

    def check_pod_travel(
        self, pod_teams: List[NCAABracketTeam], host_team: NCAABracketTeam
    ) -> int:
        """
        Returns number of necessary flights in a pod.
        A flight is needed if any team is more than 500 miles from the host.
        """
        flights = 0
        for team in pod_teams:
            if team.team != host_team.team:
                if self.get_distance(team.team, host_team.team) > 500:
                    flights += 1
        return flights

    def score_pod_diversity(
        self, pod_teams: List[NCAABracketTeam]
    ) -> Tuple[float, float]:
        """Returns (region_score, conference_score) for a pod."""
        # Ensure exactly 4 teams
        assert len(pod_teams) == 4, f"Expected 4 teams, got {len(pod_teams)}"

        regions = set(team.region for team in pod_teams)
        conferences = set(team.conference for team in pod_teams)

        # Scoring logic that rewards diversity within a 4-team pod
        region_score = {
            4: 1.5,  # All 4 regions different
            3: 1.0,  # 3 different regions
            2: 0.5,  # 2 different regions
        }.get(len(regions), 0)

        conference_score = {
            4: 1.5,  # All 4 conferences different
            3: 1.0,  # 3 different conferences
            2: 0.5,  # 2 different conferences
        }.get(len(conferences), 0)

        return region_score, conference_score

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

        # Debug print teams and their quadrant seeds
        print(f"\nTeams in {quadrant}:")
        for team in teams_by_seed:
            print(
                f"Team: {team.team}, Overall Seed: {team.overall_seed}, Quadrant Seed: {self.get_quadrant_seed(team.overall_seed)}"
            )

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

                # Debug print selected teams
                print(f"\nLooking for {pod_name} matchup:")
                print(f"Seeking high seed {high_seed}, low seed {low_seed}")
                print(
                    f"High Team: {high_team.team} (Quadrant Seed: {self.get_quadrant_seed(high_team.overall_seed)})"
                )
                print(
                    f"Low Team: {low_team.team} (Quadrant Seed: {self.get_quadrant_seed(low_team.overall_seed)})"
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

        # Group matchups by quadrant and pod
        quadrant_pods = {}
        for matchup in matchups:
            if matchup.quadrant_name not in quadrant_pods:
                quadrant_pods[matchup.quadrant_name] = {}

            pod = matchup.pod_name
            if pod not in quadrant_pods[matchup.quadrant_name]:
                quadrant_pods[matchup.quadrant_name][pod] = []

            quadrant_pods[matchup.quadrant_name][pod].extend(
                [matchup.team_a, matchup.team_b]
            )

        # Score each pod in each quadrant
        for quadrant, pods in quadrant_pods.items():
            for pod, pod_teams in pods.items():
                # Ensure we have exactly 4 teams in the pod
                if len(pod_teams) != 4:
                    raise ValueError(f"Pod {pod} in {quadrant} does not have 4 teams")

                # Look up full team details
                full_pod_teams = [self.get_team(team) for team in pod_teams]

                # Seed Difference Penalty
                for team_name in pod_teams:
                    team = self.get_team(team_name)
                    # Find the matching matchup to get the actual seed
                    matchup = next(
                        m
                        for m in matchups
                        if m.pod_name == pod
                        and (m.team_a == team_name or m.team_b == team_name)
                    )
                    seed = (
                        matchup.team_a_actual_seed
                        if team_name == matchup.team_a
                        else matchup.team_b_actual_seed
                    )

                    diff_penalty = self.calculate_seed_difference(team, seed)
                    total_score += diff_penalty

                # Host team (first team in the pod)
                host_team = full_pod_teams[0]

                # Travel Penalty
                travel_penalty = self.check_pod_travel(full_pod_teams, host_team) * 2.0
                total_score += travel_penalty

                # Diversity Bonus
                region_score, conference_score = self.score_pod_diversity(
                    full_pod_teams
                )
                total_score -= region_score + conference_score

        return total_score

    def optimize_bracket(
        self, initial_bracket: List[NCAABracketMatchup], iterations: int = 1000
    ) -> List[NCAABracketMatchup]:
        """
        Attempts to optimize bracket through iterative improvements.

        Strategy:
        1. Start with initial bracket
        2. Generate small random modifications
        3. Accept modifications that improve the score
        4. Use simulated annealing-like approach
        """
        import random

        current_bracket = initial_bracket.copy()
        current_score = self.score_bracket(current_bracket)
        best_bracket = current_bracket
        best_score = current_score

        for _ in range(iterations):
            # Create a copy of the current bracket to modify
            modified_bracket = current_bracket.copy()

            # Attempt a small modification (e.g., swap teams within a pod)
            try:
                # Select a random quadrant
                quadrants = list(
                    set(matchup.quadrant_name for matchup in modified_bracket)
                )
                selected_quadrant = random.choice(quadrants)

                # Get matchups in this quadrant
                quadrant_matchups = [
                    m for m in modified_bracket if m.quadrant_name == selected_quadrant
                ]

                # Randomly select two matchups to potentially swap
                if len(quadrant_matchups) >= 2:
                    m1, m2 = random.sample(quadrant_matchups, 2)

                    # Swap teams while preserving actual seeds
                    m1.team_a, m2.team_a = m2.team_a, m1.team_a
                    m1.team_b, m2.team_b = m2.team_b, m1.team_b

                    # Recalculate score
                    new_score = self.score_bracket(modified_bracket)

                    # Accept if score improves
                    if new_score < current_score:
                        current_bracket = modified_bracket
                        current_score = new_score

                        # Update best bracket if needed
                        if current_score < best_score:
                            best_bracket = current_bracket
                            best_score = current_score

            except Exception:
                # If modification fails, continue to next iteration
                continue

        return best_bracket

    def generate_bracket(self) -> List[NCAABracketMatchup]:
        """Main function to generate and optimize bracket."""
        initial_bracket = self.create_initial_bracket()
        # optimized_bracket = self.optimize_bracket(initial_bracket)
        # return optimized_bracket
        return initial_bracket

    def print_teams_with_quadrant_seeds(self):
        """Prints all teams with their corresponding quadrant seeds (1-16)."""

        # Calculate quadrant seed (1-16) from overall seed (1-64)
        def get_quadrant_seed(overall_seed):
            return ((overall_seed - 1) % 16) + 1

        # Sort teams by overall seed
        sorted_teams = sorted(self.teams, key=lambda x: x.overall_seed)

        print("\nTeams with Quadrant Seeds:")
        print("-" * 60)
        print(
            f"{'Overall Seed':<15} {'Quadrant Seed':<15} {'Team':<30} {'Conference':<10}"
        )
        print("-" * 60)

        for team in sorted_teams:
            quadrant_seed = get_quadrant_seed(team.overall_seed)
            print(
                f"{team.overall_seed:<15} {quadrant_seed:<15} {team.team:<30} {team.conference:<10}"
            )


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
