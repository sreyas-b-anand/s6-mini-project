import os
import joblib
import pandas as pd
from server.schema.score import ScoreRequest
from server.utils import GraphFeatureExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svm_pipeline.pkl")
GRAPH_PATH = os.path.join(BASE_DIR, "..", "models", "graph_model.pkl")

async def classical_ml(req: ScoreRequest):
    #code to load the pickle file for ML model and graph network

    try:
        g = joblib.load(GRAPH_PATH)
   
        data = joblib.load(MODEL_PATH)
        model = data["model"]
        le = data["label_encoder"]
        
        if(not req.text or not req.rating or not req.category):
            raise Exception("Missing fields")
            
    
        input_df = pd.DataFrame([{
            "category": req.category,
            "rating": req.rating,
            "text_": req.text
        }])
        graph_result = g.predict_review(
            review_text=req.text,
            rating=req.rating,
            category=req.category
        )
        
        pred = model.predict(input_df)
        decision_score = model.decision_function(input_df)
    
        final_label = le.inverse_transform(pred)
        
        
        result = "Fake" if final_label[0] == "CG" else "Valid"
        
        return {
            "prediction_ml": result,
            "prediction_g":graph_result["prediction"],
            "similar_reviews_found":graph_result["similar_reviews_found"],
            "confidence_ml": float(decision_score[0]) ,
            "confidence_g" : graph_result["suspicion_score"]
        }
    except:
        raise Exception("An error occured in the server")
        