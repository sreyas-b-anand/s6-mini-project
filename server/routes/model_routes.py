import os
from fastapi import APIRouter , status
from server.controllers import bert_model
from server.controllers import classical_ml 
from server.schema import ScoreRequest, ReviewRequest
from server.utils.graph import GraphFeatureExtractor
model_router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "..", "data", "reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "graph_model.pkl")

@model_router.post('/bert_score')
async def get_bert_score(req:ReviewRequest):
    return await bert_model(req.text)


@model_router.post('/ml_score')
async def get_ml_score(req : ScoreRequest):
    return await classical_ml(req)

# only for training graph model 
@model_router.get('/train_graph')
async def graph_training():
    g = GraphFeatureExtractor()
    g.build_reference_graph(DATASET_PATH)
    g.save_model(MODEL_PATH)
    return status.HTTP_200_OK