import os
import sys
from pathlib import Path
from fastapi import HTTPException
import joblib
import pandas as pd
import math
from schema.score import MlScoreRequest
import asyncio

# allow loading pickles created with `server.*` module path when running as `uvicorn main:app`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, ".." , "models", "svm_pipeline.pkl")
GRAPH_PATH = os.path.join(BASE_DIR, "..", "models", "graph_model.pkl")

def classify_strength(score):
    if score < 0.25:
        return "Strongly Valid"
    elif score < 0.47:
        return "Likely Valid"
    elif score < 0.65:
        return "Likely Fake"
    else:
        return "Strongly Fake"
    
async def combine_predictions(ml_confidence, graph_confidence, rating, text):
   
    ML_WEIGHT = 0.65
    GRAPH_WEIGHT = 0.35

    ml_prob = 1 / (1 + math.exp(-ml_confidence))

    graph_prob = max(0, min(1, graph_confidence))

    text_lower = text.lower()
    positive_words = ["amazing", "great", "excellent", "good", "love"]
    negative_words = ["bad", "worst", "poor", "terrible"]

    sentiment_flag = 0

    if any(word in text_lower for word in positive_words) and rating <= 2:
        sentiment_flag = 1 
    elif any(word in text_lower for word in negative_words) and rating >= 4:
        sentiment_flag = 1

    if sentiment_flag:
        graph_prob += 0.1
        graph_prob = min(graph_prob, 1)
    
    combined_score = (ml_prob * ML_WEIGHT) + (graph_prob * GRAPH_WEIGHT)
    threshold = 0.47
    final = "Fake" if combined_score >= threshold else "Valid"

    return final, combined_score

g = joblib.load(GRAPH_PATH)
data = joblib.load(MODEL_PATH)
model = data["model"]
le = data["label_encoder"]

async def classical_ml(req: MlScoreRequest):

    try:
        

        CATEGORY_MAP = {
            "Electronics": "Electronics_5",
            "Books": "Books_5",
            "Movies": "Movies_and_TV_5",
            "Home Appliances": "Home_and_Kitchen_5",
            "Sports": "Sports_and_Outdoors_5",
            "Tools and Home Improvements": "Tools_and_Home_Improvement_5",
            "Pets supplies": "Pet_Supplies_5",
            "Kindle": "Kindle_Store_5",
            "Toys": "Toys_and_Games_5",
            "Fashion and clothing": "Clothing_Shoes_and_Jewelry_5"
        }

       
        if not req.text or not req.rating or not req.category:
            raise HTTPException(status_code=400, detail="Missing fields")

        mapped_category = CATEGORY_MAP.get(req.category)

        if mapped_category is None:
            raise HTTPException(status_code=400, detail="Invalid category")

        input_df = pd.DataFrame([{
            "category": req.category,
            "rating": req.rating,
            "text_": req.text
        }])
        
        graph_task = asyncio.to_thread(
            g.predict_review,
            review_text=req.text,
            rating=req.rating,
            category=req.category
        )
        
        ml_task = asyncio.to_thread(
        lambda: (
            model.predict(input_df),
            model.decision_function(input_df)
            )
        )   
        graph_data, (pred, decision_score) = await asyncio.gather(
        graph_task, ml_task
        )
        graph_result = graph_data["prediction"]              # "Fake" / "Real"
        graph_confidence = graph_data["suspicion_score"]     # 0 -> 1
        similar_reviews = graph_data["similar_reviews_found"]


        ml_confidence = float(decision_score[0])
        final_label = le.inverse_transform(pred)

        ml_result = "Fake" if final_label[0] == "CG" else "Valid"

        final_result, final_confidence_score = await combine_predictions(
            ml_confidence,
            graph_confidence,
            req.rating,
            req.text
        )
        
        strength = classify_strength(final_confidence_score)

        return {
            "prediction_ml": ml_result,
            "prediction_g": graph_result,
            "similar_reviews_found": similar_reviews,
            "strength": strength,
            "confidence_ml": ml_confidence,
            "confidence_g": graph_confidence,
            "final_result": final_result,
            "final_confidence": final_confidence_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))