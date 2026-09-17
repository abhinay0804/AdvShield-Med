from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# --- User Schemas ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    role: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- Scan Schemas ---
class ScanResponse(BaseModel):
    id: int
    patient_name: str
    timestamp: datetime
    original_image_path: str
    heatmap_image_path: str
    gatekeeper_flag: bool
    gatekeeper_confidence: float
    diagnosis: str

    class Config:
        from_attributes = True
