# This code repository

A Python application for calculating NPI (NCAA Power Index) for Division III Women's Basketball.

NCAA Power Index (NPI) is the NCAA's new system of ranking teams and selecting them to NCAA postseason tournaments in Division III athletics.

## The app

This application is NOT currently designed to be super user-friendly for the non-programmer.

Games and teams live in their own games.txt and teams.txt files, each in year directories.

The data comes from Massey's data.

Most functions do not have parameters to control: (Sport, Gender, Year)

## Generating NPI

Like any modern python script or app, you should be running it in a [virtual environment](https://docs.python.org/3/library/venv.html).

1. Set up your virtual environment and activate it.
2. Install dependencies from `requirements.txt`
3. Run NPI: `python -m myapp.commands main`

all instructions need to be run at the command line

The command to generate NPI is currently not configurable by year. You'll have to manually edit the year in `main.py`

The weights for Win%, SOS, QWB, etc are all currently hardcoded for WBB.

