# config mgmt package

import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent

def load_config(config_name):
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_name}.yaml not found in {CONFIG_DIR}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_model_config():
    return load_config("models")

def get_training_config():
    return load_config("training")

def get_teacher_models_config(tier="free"):
    config = load_config("teacher_models")
    return config.get(f"{tier}_tier", [])

def get_teacher_models(tier="free"):
    return get_teacher_models_config(tier)