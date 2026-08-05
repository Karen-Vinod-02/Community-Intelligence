from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    description: str

class AnalyzeResponse(BaseModel):
    description: str