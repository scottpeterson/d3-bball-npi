import csv
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def massey_ratings_getter(url, year="2025"):
    """
    Scrapes team rankings data from Massey Ratings and saves to CSV
    """
    output_path = Path(__file__).parent / "data" / year
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "massey.csv"
    team_mapping_file = output_path / "teams_mapping.txt"

    try:
        if not team_mapping_file.exists():
            raise ValueError("Required mapping file not found")

        team_mappings = pd.read_csv(team_mapping_file, skipinitialspace=True)
        massey_map = dict(
            zip(team_mappings["Massey_Name"], team_mappings["Scott_Name"])
        )

        # Set up Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.page_load_strategy = "eager"

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(30)

        driver.get(url)

        wait = WebDriverWait(driver, 20)
        table = wait.until(EC.presence_of_element_located((By.ID, "tbl")))
        time.sleep(2)

        rows = driver.find_elements(By.CLASS_NAME, "bodyrow")

        # Process and save rankings
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Team", "Ranking"])

            for row in rows:
                try:
                    team_cell = row.find_element(By.CLASS_NAME, "fteam")
                    massey_name = team_cell.find_element(By.TAG_NAME, "a").text

                    # Replace whitespace with underscore for mapping
                    massey_name_underscored = massey_name.replace(" ", "_")
                    scott_name = massey_map.get(massey_name_underscored, massey_name)

                    rank_cells = row.find_elements(By.CLASS_NAME, "frank")
                    if len(rank_cells) > 1:
                        ranking = rank_cells[1].text.split("\n")[0]
                        writer.writerow([scott_name, ranking])
                except Exception as row_error:
                    continue

        return True

    except Exception as e:
        return False

    finally:
        if "driver" in locals():
            driver.quit()
