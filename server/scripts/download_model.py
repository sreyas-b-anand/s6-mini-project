import os
import gdown
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("DRIVE_LINK")

if not url:
    raise ValueError("DRIVE_LINK environment variable not set")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.listdir(MODEL_DIR):
    print("Downloading models...")
    gdown.download_folder(url, output=MODEL_DIR, quiet=False)
else:
    print("Models already exist, skipping download.")