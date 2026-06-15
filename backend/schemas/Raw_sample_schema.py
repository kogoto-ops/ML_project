from pydantic import BaseModel
from typing import Optional

class RawSampleSchema(BaseModel):
    year: int
    make: str
    model: str
    trim: str
    body: str
    transmission: str
    state: str
    condition: float
    odometer: float
    color: str
    interior: str
    saledate: str  # raw date string
