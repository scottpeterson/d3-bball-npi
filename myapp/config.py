import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    APP_NAME = os.getenv("APP_NAME", "MyApp")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
