"""
Request schemas for API validation
"""
from pydantic import BaseModel, Field
from typing import Optional


class TryOnRequest(BaseModel):
    """Request schema for virtual try-on"""
    
    category: Optional[str] = Field(
        default="upper_body",
        description="Garment category: upper_body, lower_body, or dresses"
    )
    num_inference_steps: Optional[int] = Field(
        default=30,
        ge=10,
        le=50,
        description="Number of denoising steps (10-50)"
    )
    guidance_scale: Optional[float] = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Guidance scale for generation (1.0-5.0)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "upper_body",
                "num_inference_steps": 30,
                "guidance_scale": 2.0
            }
        }
