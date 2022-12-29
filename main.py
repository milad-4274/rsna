from utils import load_jpeg2_image, plt_imshow
from pathlib import Path
import argparse
from dotenv import load_dotenv
import os
import warnings
from models import Resnet18Extractor
import torch
from data import create_dataset


ENV_SUFFIX = ".env"

parser = argparse.ArgumentParser()

parser.add_argument("env",default="local")


args = parser.parse_args()
env_path = Path(args.env).with_suffix(ENV_SUFFIX)

envloaded = load_dotenv(env_path)

if envloaded:
    print(f"environment {env_path} loaded successfully")
else:
    warnings.warn(f"no environment found in path {env_path}")

data_dir = os.getenv("DATA_DIR")
model_name = os.getenv("MODEL")
print(data_dir)


model = Resnet18Extractor()

size_transform = model.get_transform()

dataset = create_dataset(path, transform, augment)


