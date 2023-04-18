import json
import os
import shutil
import subprocess
import sys
import zipfile

import yaml

KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
KAGGLE_CONFIG_FILE = f"{KAGGLE_CONFIG_DIR}/kaggle.json"


def config_kaggle():
    if os.path.isfile(KAGGLE_CONFIG_FILE):
        return
    else:
        if not os.path.isfile('conf/local/credentials.yml'):
            print(repr("Not Found: 'conf/local/credentials.yml'"))
            return
        with open('conf/local/credentials.yml', 'r') as file:
            credentials = yaml.safe_load(file)
            init_on_kaggle(
                credentials['kaggle']['username'], credentials['kaggle']['key'])


def init_on_kaggle(username, api_key):
    os.makedirs(KAGGLE_CONFIG_DIR, exist_ok=True)
    api_dict = {"username": username, "key": api_key}
    with open(KAGGLE_CONFIG_FILE, "w", encoding='utf-8') as f:
        json.dump(api_dict, f)

    cmd = f"chmod 600 {KAGGLE_CONFIG_FILE}"
    output = subprocess.check_output(cmd.split(" "))
    output = output.decode(encoding='UTF-8')
    print(f"Generate kaggle.json:\n\t{KAGGLE_CONFIG_FILE}")


def get_kobe_bryant_shot_selection(dir: str = "Data/"):
    import kaggle

    competition_name = "kobe-bryant-shot-selection"
    try:
        kaggle.api.competition_download_files(
            competition_name, path=f"{dir}download")

        with zipfile.ZipFile(f"{dir}download/{competition_name}.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{dir}download/")

        with zipfile.ZipFile(f"{dir}download/data.csv.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{dir}")

    except FileNotFoundError:
        print(repr("Downloaded files don`t found"))
    except kaggle.rest.ApiException:
        print(repr("Kaggle Api Error"))

    success = os.path.isfile(f"{dir}data.csv")

    if success:
        os.rename(f"{dir}data.csv", f"{dir}kobe_dataset.csv",)
        print("Data received with success")
    else:
        print("Don`t received data.cvs")

    if os.path.exists(f"{dir}download/"):
        shutil.rmtree(f"{dir}download/")

    return success


config_kaggle()
get_kobe_bryant_shot_selection()
