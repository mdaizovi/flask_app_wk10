import os

settings_file = os.path.realpath(__file__)
BASE_DIR = settings_file.replace("/settings.py", "")
DATA_DIR = settings_file.replace("settings.py", "data/")
