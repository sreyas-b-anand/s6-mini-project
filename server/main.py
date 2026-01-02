from fastapi import FastAPI
from .routes.model_routes import model_router

app = FastAPI()

app.include_router(model_router , prefix="/model")

@app.get("/")
async def root():
    return {"message": "Hello World"}