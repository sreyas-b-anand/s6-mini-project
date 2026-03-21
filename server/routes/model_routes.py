import os
from fastapi import APIRouter , status , HTTPException
from server.controllers import bert_model
from server.controllers import classical_ml 
from server.schema import BertScoreRequest , MlScoreRequest
from server.utils import GraphFeatureExtractor
from server.utils import scrape_and_save
model_router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "graph_model.pkl")

@model_router.post('/bert_score')
async def get_bert_score(req: BertScoreRequest):

    if req.type == "link":
        if not req.url:
            raise HTTPException(status_code=400, detail="url is required for type 'link'.")
        
        scraped = await scrape_and_save(req.url)   # scrap fn
        texts   = [r["text"] for r in scraped]
        ratings = [r["rating"] for r in scraped]

    elif req.type == "single":
        if not req.review or req.rating is None:
            raise HTTPException(status_code=400, detail="review and rating are required for type 'single'.")
        
        texts   = [req.review]      
        ratings = [req.rating]      

    else:
        raise HTTPException(status_code=400, detail="type must be 'link' or 'single'.")

    return await bert_model(texts, ratings)

@model_router.post('/ml_score')
async def get_ml_score(req : MlScoreRequest):
    return await classical_ml(req)

# only for training graph model 
@model_router.get('/train_graph')
async def graph_training():
    g = GraphFeatureExtractor()
    g.build_reference_graph(DATASET_PATH)
    g.save_model(MODEL_PATH)
    return status.HTTP_200_OK