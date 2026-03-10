from pydantic import BaseModel

class ScoreRequest(BaseModel):
    category : str
    rating : float
    text : str

class ReviewRequest(BaseModel):
    text: str