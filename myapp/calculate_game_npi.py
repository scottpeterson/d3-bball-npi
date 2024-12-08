def calculate_game_npi(won, opponent_npi):
    win_component = 100 if won else 0
    base_npi = (win_component * 0.20) + (opponent_npi * 0.80)

    quality_bonus = max(0, (opponent_npi - 54.00) * (2 / 3)) if won else 0

    total_npi = base_npi + quality_bonus
    return total_npi