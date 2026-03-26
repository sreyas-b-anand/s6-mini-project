import os
from utils.bert.bert_inference import FakeReviewDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# __file__ = server/controllers/your_controller.py
# BASE_DIR = server/

detector = FakeReviewDetector(
    model_dir=os.path.join(BASE_DIR, "models", "bert_model")  
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
            "rating" : rating,
            "prediction": r["label"], 
            "confidence": r["confidence"]
        }
        for text , rating , r in zip(texts , ratings, results)
    ]
    
    fake_count = sum(1 for r in results if r["label"] == "CG")
    total = len(results)
    fake_percentage = (fake_count / total) * 100
    
    return {
        "results": formatted_results,
        "fake_percentage": round(fake_percentage, 2), 
        "total_reviews": total,
    }