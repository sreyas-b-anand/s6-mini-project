import os
from server.utils.bert.bert_inference import FakeReviewDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__ = server/controllers/your_controller.py
# BASE_DIR = server/

detector = FakeReviewDetector(
    model_dir=os.path.join(BASE_DIR, "models", "bert_model_new")  
)

async def bert_model(texts: str, ratings: float):
    results = detector.predict(texts=texts, ratings=ratings)
    if len(texts) != len(results):
        raise ValueError("Length mismatch in texts")
    
    if len(results) == 1:
        r = results[0]
        return {
            "prediction": r["label"],
            "confidence": r["confidence"]
        }
    
    formatted_results = [
        {
            "text": text,
            "prediction": r["label"], 
            "confidence": r["confidence"]
        }
        for text, r in zip(texts, results)
    ]
    
    return formatted_results