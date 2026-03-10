import os
import gdown
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("DRIVE_LINK")

if not url:
    raise ValueError("DRIVE_LINK environment variable not set")

output_path = "../models"

gdown.download_folder(url, output=output_path, quiet=False)