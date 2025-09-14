from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class SessionCreate(BaseModel):
    title: str = "Untitled Session"


class Segment(BaseModel):
    sessionId: str
    idx: int
    tStart: float
    tEnd: float
    lang: str
    speaker: str
    textSrc: str
    textEn: str
    partial: bool = False
    confidence: float


class Session(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str = Field(alias="_id")
    title: str
    createdAt: datetime
    status: str
    updatedAt: Optional[datetime] = None
    segments: List[Segment] = []
