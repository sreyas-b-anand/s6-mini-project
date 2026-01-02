from fastapi import APIRouter
from ..models.bert_model import bert_model
from ..models.classical_ml import classical_ml 
model_router = APIRouter()

@model_router.get('/bert_score')
async def get_bert_score():
    return await bert_model()


@model_router.get('/ml_score')
async def get_ml_score():
    return await classical_ml()