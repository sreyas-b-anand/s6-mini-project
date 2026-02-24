import os
from server.utils.graph import visualize_review_graph
from server.utils.graph import GraphFeatureExtractor
import joblib
import pandas as pd
from server.schema.score import ScoreRequest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "svm_pipeline.pkl")

async def classical_ml(req: ScoreRequest):
    #code to load the pickle file for ML model
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    le = data["label_encoder"]

    input_df = pd.DataFrame([{
        "category": req.category,
        "rating": req.rating,
        "text_": req.text
    }])
    
    pred = model.predict(input_df)
    decision_score = model.decision_function(input_df)

    final_label = le.inverse_transform(pred)
    
    
    result = "Fake" if final_label[0] == "CG" else "Valid"
    
    return {
        "prediction": result,
        "confidence": float(decision_score[0])
    }