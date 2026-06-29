import os
import sys
from pathlib import Path

# Ensure project root is on Python path so pickled models referencing server.* can be loaded
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI , status
from fastapi.middleware.cors import CORSMiddleware
from routes import model_router
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

origins = [
    os.getenv("FRONTEND_URL_1"),
    os.getenv("FRONTEND_URL_2"),
    os.getenv("FRONTEND_URL_3", "")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],          
)

app.include_router(model_router, prefix="/model")


@app.get("/ping")
async def root():
    return status.HTTP_200_OK