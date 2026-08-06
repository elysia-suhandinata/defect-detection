from pydantic import BaseModel
from typing import List, Optional


class DefectPrediction(BaseModel):
    defect_class: str
    probability: float
    detected: bool


class SegmentationResult(BaseModel):
    available: bool = False
    location: Optional[str] = None
    defect_area_percent: Optional[float] = None


class InspectionResult(BaseModel):
    filename: str
    predictions: List[DefectPrediction]
    segmentation: SegmentationResult
    inspection_report: Optional[str] = None