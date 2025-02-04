#!/bin/bash
cd /Users/scottpeterson/projects/npi
source /Users/scottpeterson/projects/npi/venv/bin/activate
/Users/scottpeterson/projects/npi/venv/bin/python -m myapp.commands get_games
/Users/scottpeterson/projects/npi/venv/bin/python -m myapp.commands main
/Users/scottpeterson/projects/npi/venv/bin/python -m myapp.commands bidirectional
/Users/scottpeterson/projects/npi/venv/bin/python -m myapp.commands massey
/Users/scottpeterson/projects/npi/venv/bin/python -m myapp.commands eff
/Users/scottpeterson/projects/npi/venv/bin/python -m myapp.commands run_multiple
deactivate