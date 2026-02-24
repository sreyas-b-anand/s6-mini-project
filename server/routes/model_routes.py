from fastapi import APIRouter
from ..controllers.bert_model import bert_model
from ..controllers.classical_ml import classical_ml 
from ..schema import ScoreRequest
model_router = APIRouter()

@model_router.get('/bert_score')
async def get_bert_score():
    return await bert_model()


@model_router.post('/ml_score')
async def get_ml_score(req : ScoreRequest):
    return await classical_ml(req)