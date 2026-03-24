import os
import joblib
import pandas as pd
import math
from server.schema.score import MlScoreRequest
import asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svm_pipeline.pkl")
GRAPH_PATH = os.path.join(BASE_DIR, "..", "models", "graph_model.pkl")

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
            raise Exception("Missing fields")

        mapped_category = CATEGORY_MAP.get(req.category)

        if mapped_category is None:
            raise Exception("Invalid category")

        input_df = pd.DataFrame([{
            "category": req.category,
            "rating": req.rating,
            "text_": req.text
        }])

        # graph_data = g.predict_review(
        #     review_text=req.text,
        #     rating=req.rating,
        #     category=req.category
        # )
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

        return {
            "prediction_ml": ml_result,
            "prediction_g": graph_result,
            "similar_reviews_found": similar_reviews,
            "confidence_ml": ml_confidence,
            "confidence_g": graph_confidence,
            "final_result": final_result,
            "final_confidence": final_confidence_score
        }

    except Exception as e:
        raise Exception(e)