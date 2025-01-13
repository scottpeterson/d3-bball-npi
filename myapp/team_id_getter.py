import requests
from pathlib import Path
from typing import Dict, Set, Tuple

def parse_massey_teams(response_text: str) -> Dict[str, str]:
    """
    Parse Massey Ratings response text into a dictionary of ID to name mappings.
    
    Args:
        response_text: Raw text from Massey Ratings
        
    Returns:
        Dictionary mapping IDs to team names
    """
    massey_teams = {}
    
    for line in response_text.strip().split('\n'):
        try:
            # Split by comma and strip whitespace
            id_str, name = line.split(',', 1)
            team_id = id_str.strip()
            team_name = name.strip()
            massey_teams[team_id] = team_name
        except ValueError:
            print(f"Warning: Could not parse line: {line}")
            continue
            
    return massey_teams

def load_existing_mappings(path: Path) -> Dict[str, str]:
    """
    Load existing team mappings from teams_mapping.txt
    
    Args:
        base_path: Base path to data directory
        year: Year to load mappings for
        
    Returns:
        Dictionary mapping IDs to team names
    """
    existing_teams = {}
    mapping_file = path / "teams.txt"
    
    try:
        with open(mapping_file, 'r') as f:
            for line in f:
                try:
                    team_id, team_name = line.strip().split(',', 1)
                    existing_teams[team_id.strip()] = team_name.strip()
                except ValueError:
                    print(f"Warning: Could not parse mapping line: {line}")
                    continue
    except FileNotFoundError:
        print(f"Error: Could not find mapping file at {mapping_file}")
        
    return existing_teams

def compare_team_mappings(massey_teams: Dict[str, str], 
                         existing_teams: Dict[str, str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Compare Massey teams with existing mappings.
    
    Returns:
        Tuple of (new_teams, missing_teams, differing_names)
    """
    massey_ids = set(massey_teams.keys())
    existing_ids = set(existing_teams.keys())
    
    # Find teams in Massey but not in existing
    new_teams = massey_ids - existing_ids
    
    # Find teams in existing but not in Massey
    missing_teams = existing_ids - massey_ids
    
    # Find teams with different names
    differing_names = set()
    for team_id in massey_ids & existing_ids:
        if massey_teams[team_id] != existing_teams[team_id]:
            differing_names.add(team_id)
            
    return new_teams, missing_teams, differing_names

def team_ids_getter(url: str, year: str = "2025") -> bool:
    # Set up paths relative to this file
    base_path = Path(__file__).parent / "data"
    output_path = base_path / year
    """
    Get team IDs from Massey Ratings and compare with existing mappings.
    
    Args:
        url: Massey Ratings URL
        year: Year to process
        base_path: Base path to data directory
    """
    try:
        # Get Massey data
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse Massey teams
        massey_teams = parse_massey_teams(response.text)
        print(f"Loaded {len(massey_teams)} teams from Massey Ratings")
        
        # Load existing mappings
        existing_teams = load_existing_mappings(output_path)
        print(f"Loaded {len(existing_teams)} teams from existing mappings")
        
        # Compare mappings
        new_teams, missing_teams, differing_names = compare_team_mappings(
            massey_teams, existing_teams
        )
        
        # Report results
        print("\nComparison Results:")
        print(f"Teams in Massey but not in mappings ({len(new_teams)}):")
        for team_id in sorted(new_teams):
            print(f"  {team_id}: {massey_teams[team_id]}")
            
        print(f"\nTeams in mappings but not in Massey ({len(missing_teams)}):")
        for team_id in sorted(missing_teams):
            print(f"  {team_id}: {existing_teams[team_id]}")
            
        print(f"\nTeams with different names ({len(differing_names)}):")
        for team_id in sorted(differing_names):
            print(f"  {team_id}: {existing_teams[team_id]} vs {massey_teams[team_id]}")
            
        return True
        
    except (requests.RequestException, IOError) as e:
        print(f"Error: {e}")
        return False