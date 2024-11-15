import sys
from pathlib import Path
import csv
from .main import load_teams, process_games_bidirectional

def run_bidirectional():
    """Entry point for bidirectional games processing."""

    # Get year from command line or use default
    if len(sys.argv) > 2:
        year = sys.argv[2]
    else:
        year = "2025"  # Default year
        
    print(f"Processing data for year {year}")
    
    # Create proper Path objects
    base_path = Path(__file__).parent / 'data'
    
    # Load teams and games from year-specific directories
    valid_teams = load_teams(base_path, year)
    games_path = base_path / year / 'games.txt'
    results = process_games_bidirectional(games_path, valid_teams)
    
    if not results:
        print("No valid results to write")
        return
    
    # Find the newest date from the results
    newest_date = ""
    for result in results:
        date = result.split(',')[0]  # Get date from first column
        if date > newest_date:
            newest_date = date
    
    # Convert date format from mm/dd/yyyy to mmddyyyy for filename
    month, day, year_part = newest_date.split('/')
    filename_date = f"{month}{day}{year_part}"
    
    # Create output filename
    output_filename = f"{filename_date}_WBB_results.csv"
    output_path = base_path / year / output_filename
    
    # Ensure the year directory exists
    (base_path / year).mkdir(exist_ok=True)
    
    # Write results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        # Write all results with updated header
        csvfile.write("Date,Team A,Team B,Home/Away,Result\n")  # Updated header
        for result in results:
            csvfile.write(f"{result}\n")
    
    print(f"Results written to: {output_filename}")
    print(f"Total games processed: {len(results)//2}")

def run_main():
    """Wrapper for the original main function."""
    from .main import main
    main()

if __name__ == '__main__':
    import sys
    
    commands = {
        'bidirectional': run_bidirectional,
        'main': run_main
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Available commands:")
        for cmd in commands:
            print(f"  - {cmd}")
        print("\nUsage:")
        print("  bidirectional [year]  - Process games for specified year (default: 2024)")
        sys.exit(1)
        
    commands[sys.argv[1]]()