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
    return [
        {"prediction": r["label"], "confidence": r["confidence"]}
        for r in results
    ]