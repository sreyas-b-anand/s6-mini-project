from pydantic import BaseModel
from typing import Optional
class MlScoreRequest(BaseModel):
    category : str
    rating : float
    text : str

class BertScoreRequest(BaseModel):
    type: str      
    url: Optional[str] = None
    review: Optional[str] = None
    rating: Optional[float] = None       