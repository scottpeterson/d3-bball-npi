import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DataLoader:
    @staticmethod
    def load_teams(file_path):
        """Load team names and seeds from a file."""
        teams = []
        seeds = {}
        with open(file_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    team = parts[0].strip()
                    # Remove trailing comma if present
                    if team.endswith(","):
                        team = team[:-1]
                    teams.append(team)
                    # Check if seed is available
                    if len(parts) >= 2:
                        try:
                            seed = int(parts[1].strip())
                            seeds[team] = seed
                        except (ValueError, IndexError):
                            seeds[team] = 999  # Default high seed if not available
        return teams, seeds

    @staticmethod
    def load_distances(file_path):
        """Load distances between teams from a file."""
        distances = {}
        first_line = True
        with open(file_path, "r") as f:
            for line in f.readlines():
                # Skip header row
                if first_line:
                    first_line = False
                    continue

                parts = line.strip().split(",")
                if len(parts) >= 3:
                    try:
                        team1 = parts[0].strip()
                        team2 = parts[1].strip()
                        distance = float(parts[2].strip())

                        # Remove trailing commas if present
                        if team1.endswith(","):
                            team1 = team1[:-1]
                        if team2.endswith(","):
                            team2 = team2[:-1]

                        # Store distance in both directions
                        if team1 not in distances:
                            distances[team1] = {}
                        if team2 not in distances:
                            distances[team2] = {}

                        distances[team1][team2] = distance
                        distances[team2][team1] = distance
                    except ValueError as e:
                        print(f"Warning: Could not parse line: {line.strip()} - {e}")

        return distances


def analyze_host_proximity(
    base_path, year, proximity_threshold=500, excluded_hosts=None
):
    """
    Analyze which teams can be potential hosts based on proximity to other teams.
    For each quadrant, analyze all possible combinations of 4 teams (one from each pair)
    and determine the best host based on proximity and seed.

    Args:
        base_path: Path to the base directory
        year: Year for analysis
        proximity_threshold: Distance threshold in miles (default 500)
        excluded_hosts: List of teams that cannot be hosts (default None)

    Returns:
        DataFrame with best host for each possible team combination
        DataFrame with statistics on team hosting percentages
    """
    # Initialize excluded hosts if None
    if excluded_hosts is None:
        excluded_hosts = []
    year_path = Path(base_path) / year
    teams, seeds = DataLoader.load_teams(year_path / "ncaa_tms.txt")
    distances = DataLoader.load_distances(year_path / "team_distances.txt")

    print(f"Loaded {len(teams)} teams")
    print(f"Loaded distance data for {len(distances)} teams")

    # Create pairs (assuming teams are listed in order)
    n_teams = len(teams)
    pairs = []
    for i in range(0, n_teams, 2):
        end_idx = min(i + 2, n_teams)
        pairs.append(teams[i:end_idx])

    # Create quadrants (groups of 8 teams, 4 pairs per quadrant)
    quadrants = []
    for i in range(0, len(pairs), 4):
        end_idx = min(i + 4, len(pairs))
        quadrant_pairs = pairs[i:end_idx]
        quadrants.append(quadrant_pairs)

    # Print quadrant structure for verification
    print(f"\nQuadrant structure:")
    for q_idx, quadrant in enumerate(quadrants):
        print(f"Quadrant {q_idx+1}: {len(quadrant)} pairs")
        for p_idx, pair in enumerate(quadrant):
            print(f"  Pair {p_idx+1}: {pair}")

    # Analyze all possible combinations within each quadrant
    results = []

    # Track statistics: how many times each team appears in combinations and hosts
    team_appearances = {team: 0 for team in teams}
    team_hosts = {team: 0 for team in teams}

    for q_idx, quadrant in enumerate(quadrants):
        print(f"\nAnalyzing Quadrant {q_idx+1}")

        # Skip incomplete quadrants
        if len(quadrant) < 4:
            print(
                f"  Skipping quadrant {q_idx+1} - insufficient pairs ({len(quadrant)})"
            )
            continue

        # Generate all possible team combinations (one from each pair)
        team_combinations = list(itertools.product(*quadrant))
        print(f"  Analyzing {len(team_combinations)} possible team combinations")

        for combo in team_combinations:
            # Update appearance count for each team in the combination
            for team in combo:
                team_appearances[team] += 1

            # For each team, count how many other teams in the combination are within proximity
            proximity_counts = {}
            for team in combo:
                count = 0
                for other_team in combo:
                    if team == other_team:
                        continue

                    # Check if distance data is available
                    if team in distances and other_team in distances[team]:
                        distance = distances[team][other_team]
                        if distance <= proximity_threshold:
                            count += 1

                proximity_counts[team] = count

            # Find the team(s) with the highest proximity count, excluding teams that cannot host
            filtered_proximity_counts = {
                team: count
                for team, count in proximity_counts.items()
                if team not in excluded_hosts
            }

            if not filtered_proximity_counts:
                # If all teams are excluded from hosting, we'll need to use the original counts
                # but this should be a rare case
                max_count = max(proximity_counts.values()) if proximity_counts else 0
                best_hosts = [
                    team
                    for team, count in proximity_counts.items()
                    if count == max_count
                ]
            else:
                max_count = max(filtered_proximity_counts.values())
                best_hosts = [
                    team
                    for team, count in filtered_proximity_counts.items()
                    if count == max_count
                ]

            # If there's a tie, select the team with the highest seed (lowest seed number)
            if len(best_hosts) > 1:
                best_hosts.sort(key=lambda t: seeds.get(t, 999))

            best_host = best_hosts[0] if best_hosts else None

            # Update host count for the selected host
            if best_host:
                team_hosts[best_host] += 1

            # Add result
            results.append(
                {
                    "quadrant": f"Quadrant {q_idx+1}",
                    "teams": ", ".join(combo),
                    "best_host": best_host,
                    "teams_within_proximity": (
                        proximity_counts.get(best_host, 0) if best_host else 0
                    ),
                    "is_tie": len(best_hosts) > 1,
                    "tie_broken_by_seed": len(best_hosts) > 1,
                    "other_potential_hosts": ", ".join([h for h in best_hosts[1:]]),
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate host percentages when teams appear in combinations
    host_percentages = {}
    for team in teams:
        if team_appearances[team] > 0:
            host_percentages[team] = (team_hosts[team] / team_appearances[team]) * 100
        else:
            host_percentages[team] = 0

    # Create a DataFrame for the statistics
    stats_data = []
    for team in teams:
        stats_data.append(
            {
                "team": team,
                "appearances": team_appearances[team],
                "times_hosting": team_hosts[team],
                "hosting_percentage": host_percentages[team],
            }
        )

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values("hosting_percentage", ascending=False)

    # Print top hosting percentages
    print("\nTop teams by hosting percentage (when they advance):")
    top_stats = stats_df[stats_df["appearances"] > 0].head(20)
    for _, row in top_stats.iterrows():
        print(
            f"{row['team']}: Hosts {row['hosting_percentage']:.1f}% of the time when in the final four"
        )

    return results_df, stats_df


def visualize_host_frequency(results_df, output_path=None):
    """
    Visualize the frequency of teams being selected as hosts.

    Args:
        results_df: DataFrame with best host for each possible team combination
        output_path: Path to save the visualization
    """
    if results_df.empty:
        print("No data to visualize")
        return

    # Count host frequency
    host_counts = results_df["best_host"].value_counts().reset_index()
    host_counts.columns = ["Team", "Frequency"]

    # Sort by frequency
    host_counts = host_counts.sort_values("Frequency", ascending=False)

    # Take top 20 teams for better visualization
    top_hosts = host_counts.head(20)

    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x="Frequency", y="Team", data=top_hosts)

    # Add frequency labels to bars
    for i, bar in enumerate(bars.patches):
        plt.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.0f}",
            ha="left",
            va="center",
        )

    plt.title("Top 20 Teams by Host Selection Frequency")
    plt.xlabel("Number of Combinations")
    plt.ylabel("Team")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    plt.show()


def visualize_host_percentage(stats_df, output_path=None):
    """
    Visualize the percentage of times a team hosts when it's part of a combination.

    Args:
        stats_df: DataFrame with hosting statistics
        output_path: Path to save the visualization
    """
    if stats_df.empty:
        print("No data to visualize")
        return

    # Filter teams that appear in at least one combination
    filtered_stats = stats_df[stats_df["appearances"] > 0].copy()

    # Sort by hosting percentage
    filtered_stats = filtered_stats.sort_values("hosting_percentage", ascending=False)

    # Take top 20 teams for better visualization
    top_stats = filtered_stats.head(20)

    plt.figure(figsize=(12, 8))
    bars = sns.barplot(x="hosting_percentage", y="team", data=top_stats)

    # Add percentage labels to bars
    for i, bar in enumerate(bars.patches):
        plt.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.1f}%",
            ha="left",
            va="center",
        )

    plt.title("Top 20 Teams by Hosting Percentage When They Advance")
    plt.xlabel("Hosting Percentage (%)")
    plt.ylabel("Team")
    plt.xlim(0, 105)  # Set x-axis limit to accommodate labels
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)

    plt.show()


def run_host_analysis():
    """
    Analyze which NCAA bracket teams can potentially host others based on proximity.
    Examines all possible combinations of teams from different pairs within a quadrant.
    """
    year = "2025"  # Hardcoded year like in your other functions
    base_path = Path(__file__).parent / "data"

    # Define teams that cannot host
    excluded_hosts = [
        "NYU",
        "Illinois Wesleyan",
        "Washington & Jefferson",
        "Elizabethtown",
    ]

    print(f"\nAnalyzing host proximity for NCAA teams {year}")
    print(f"Teams excluded from hosting: {', '.join(excluded_hosts)}")
    print("-" * 50)

    try:
        results_df, stats_df = analyze_host_proximity(
            base_path, year, excluded_hosts=excluded_hosts
        )

        # Create output directory
        output_dir = base_path / year
        output_dir.mkdir(exist_ok=True)

        # Save detailed results to CSV
        csv_path = output_dir / "host_analysis.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nDetailed host analysis saved to: {csv_path}")

        # Save statistics to CSV
        stats_path = output_dir / "host_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"Host statistics saved to: {stats_path}")

        # Create visualization
        vis_path = output_dir / "host_frequency.png"
        visualize_host_frequency(results_df, vis_path)
        print(f"Host frequency visualization saved to: {vis_path}")

        # Create hosting percentage visualization
        pct_vis_path = output_dir / "host_percentage.png"
        visualize_host_percentage(stats_df, pct_vis_path)
        print(f"Host percentage visualization saved to: {pct_vis_path}")

        return True
    except Exception as e:
        print(f"Error in host analysis: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_host_analysis()
