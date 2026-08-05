from fastapi import APIRouter

from app.models.analysis import AnalyzeRequest, AnalyzeResponse

router = APIRouter()

@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    return AnalyzeResponse(
        description=request.description
    )