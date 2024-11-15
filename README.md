# This code repository

A Python application for calculating NPI (NCAA Power Index) for Division III Women's Basketball.

NCAA Power Index (NPI) is the NCAA's new system of ranking teams and selecting them to NCAA postseason tournaments in Division III athletics.

## The app

This application is NOT currently designed to be super user-friendly for the non-programmer.

Games and teams live in their own games.txt and teams.txt files, each in year directories.

The data comes from Massey's data and needs to be manually uploaded (working on changing that).

## Generating NPI

Like any modern python script or app, you should be running it in a [virtual environment](https://docs.python.org/3/library/venv.html).

1. Set up your virtual environment and activate it.
2. Install dependencies from `requirements.txt`
3. Run NPI: `python -m myapp.commands main`

all instructions need to be run at the command line

The command to generate NPI is currently not configurable by year. You'll have to manually edit the year in `main.py`

The weights for Win%, SOS, QWB, etc are all currently hardcoded for WBB.

## Ideas for future development

- Programmatic fetching of Massey data via command. I don't want to set up automatic fetching on some cadence, but cutting out the manual copy/paste step would make the app more resilient.
- Running of commands at the command line should take in a year parameter
- Weights/dials being used should be variabilized, so the user can configure by sport. Though unsure if this code is flexible enough to work for all sports. It definitely is not set up to calculate MBB NPI, due to the different weighting of Home/Away games and Conference/Non-Conference games.