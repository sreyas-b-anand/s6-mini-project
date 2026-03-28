import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import model_router
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

origins = [
    os.getenv("FRONTEND_URL_1"),
    os.getenv("FRONTEND_URL_2"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # allowed frontend origins
    allow_credentials=True,
    allow_methods=["*"],          # allow all methods
    allow_headers=["*"],          # allow all headers
)

app.include_router(model_router, prefix="/model")


@app.get("/")
async def root():
    return {"message": "Hello World"}