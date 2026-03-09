from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.routes import model_router

app = FastAPI()

origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
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