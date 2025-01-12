def load_games(base_path, year, valid_teams):
    games = []
    seen_games = set()
    year_str = str(year)
    games_path = base_path / year_str / "games.txt"
    # games_path = base_path / year_str / "season_results.txt"
    zero_zero_count = 0
    duplicate_count = 0
    skipped_due_to_invalid_teams = 0

    try:
        with open(games_path, "r") as file:
            for line in file:
                try:
                    cols = line.strip().split(",")
                    if len(cols) < 8:
                        continue

                    # Extract game data
                    date = cols[0].strip()
                    team1_id = cols[2].strip()
                    team2_id = cols[5].strip()
                    team1_score = int(cols[4])
                    team2_score = int(cols[7])

                    # Skip 0-0 games
                    if team1_score == 0 and team2_score == 0:
                        zero_zero_count += 1
                        continue

                    # Skip games where either team is not in the valid_teams dictionary
                    if team1_id not in valid_teams or team2_id not in valid_teams:
                        skipped_due_to_invalid_teams += 1
                        continue

                    # Create a unique game identifier (ordered team IDs + date)
                    game_id = tuple(sorted([team1_id, team2_id]) + [date])

                    # Skip duplicates
                    if game_id in seen_games:
                        duplicate_count += 1
                        continue

                    seen_games.add(game_id)
                    games.append(
                        {
                            "date": date,
                            "team1_id": team1_id,
                            "team2_id": team2_id,
                            "team1_score": team1_score,
                            "team2_score": team2_score,
                        }
                    )

                except Exception as e:
                    continue

        print(f"Game Loading Statistics:")
        print(f"Total games loaded: {len(games)}")
        print(f"Skipped 0-0 games: {zero_zero_count}")
        print(f"Skipped duplicates: {duplicate_count}")
        print(f"Skipped due to invalid teams: {skipped_due_to_invalid_teams}")

    except Exception as e:
        print(f"Error loading games: {e}")

    return games
