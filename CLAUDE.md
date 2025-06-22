# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python application for calculating NPI (NCAA Power Index) for Division III Women's Basketball. The NPI is the NCAA's system for ranking teams and selecting them for postseason tournaments in Division III athletics.

## Development Commands

### Environment Setup
```bash
# Set up virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Main NPI calculation (currently hardcoded to year 2025)
python -m myapp.commands main

# Available commands (run without args to see full list):
python -m myapp.commands
```

### Key Commands
- `python -m myapp.commands main` - Calculate NPI rankings
- `python -m myapp.commands get_games` - Fetch game data from Massey
- `python -m myapp.commands bidirectional [year]` - Process games bidirectionally
- `python -m myapp.commands run_multiple` - Run multiple season simulations
- `python -m myapp.commands simulate_season` - Simulate remaining games
- `python -m myapp.commands generate_bracket` - Generate NCAA tournament bracket
- `python -m myapp.commands simulate_tournament` - Simulate tournament games

### Code Quality
```bash
# Format code
black .
isort .

# Pre-commit hooks (install once)
pre-commit install
pre-commit run --all-files
```

### Testing
The project includes pytest in requirements.txt but no test files are currently present in the codebase.

## Architecture

### Core Components
- **main.py**: Entry point for NPI calculation with iterative processing (30-80 iterations)
- **commands.py**: Command-line interface with all available operations
- **load_teams.py/load_games.py**: Data loading from text files in `data/{year}/` directories
- **process_games_iteration.py**: Core NPI calculation logic per iteration
- **simulation.py**: Game prediction and season simulation functionality
- **ncaa_bracket.py**: Tournament bracket generation and scoring

### Data Structure
- Game and team data organized by year in `myapp/data/{year}/` directories
- Main data files: `games.txt`, `teams.txt`, `conferences.txt`, etc.
- External data sources: Massey ratings, D3 Data Cast efficiency ratings
- Output: CSV files with NPI results, simulation statistics, brackets

### Key Patterns
- Iterative NPI calculation using opponent NPI values from previous iteration
- Web scraping utilities for external data sources (games_getter, massey_ratings_getter, etc.)
- Extensive simulation capabilities with configurable parameters
- CSV-based data persistence and reporting

## Configuration
- Year is currently hardcoded in multiple places (main.py:14, various commands)
- NPI weights are hardcoded for Women's Basketball
- Simulation parameters (iterations, number of simulations) are configurable in code
- External URLs for data sources are hardcoded in respective getter functions