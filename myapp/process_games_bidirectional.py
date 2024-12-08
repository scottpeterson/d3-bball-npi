def process_games_bidirectional(file_path, teams_dict):
    results = []
    skipped_games = 0
    processed_games = 0
    
    try:
        with open(file_path, "r") as file:
            for line in file:
                try:
                    cols = line.strip().split(",")
                    if len(cols) < 8:
                        continue
                        
                    team1_id = cols[2].strip()
                    team2_id = cols[5].strip()
                    
                    if team1_id not in teams_dict or team2_id not in teams_dict:
                        skipped_games += 1
                        continue
                        
                    date_str = cols[1].strip()
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    formatted_date = f"{month}/{day}/{year}"
                    
                    team1_score = int(cols[4])
                    team2_score = int(cols[7])
                    team1_location = int(cols[3].strip())
                    team2_location = int(cols[5].strip())
                    team1_name = teams_dict[team1_id]
                    team2_name = teams_dict[team2_id]
                    
                    if team1_score > team2_score:
                        team1_result = "W"
                        team2_result = "L"
                    elif team2_score > team1_score:
                        team1_result = "L"
                        team2_result = "W"
                    else:
                        team1_result = team2_result = "T"
                    
                    # Determine locations based on both indicators
                    team1_status = "Neutral" if team1_location == 0 else ("Home" if team1_location == 1 else "Road")
                    team2_status = "Neutral" if team2_location == 0 else ("Home" if team2_location == 1 else "Road")
                    
                    results.append(
                        f"{formatted_date},{team1_name},{team2_name},{team1_status},{team1_result}"
                    )
                    results.append(
                        f"{formatted_date},{team2_name},{team1_name},{team2_status},{team2_result}"
                    )
                    
                    processed_games += 1
                except Exception as e:
                    skipped_games += 1
                    continue
                    
        print(f"Processed {processed_games} games into {len(results)} bidirectional results")
        print(f"Skipped {skipped_games} games due to missing teams or invalid data")
    except Exception as e:
        print(f"Error processing games file: {e}")
        
    return results