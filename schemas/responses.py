"""
Response schemas for API
"""
from pydantic import BaseModel, Field
from typing import Optional


class TryOnResponse(BaseModel):
    """Response schema for successful try-on"""
    
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(description="Response message")
    result_image: str = Field(description="Base64 encoded result image")
    processing_time: float = Field(description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Virtual try-on completed successfully",
                "result_image": "base64_encoded_image_data...",
                "processing_time": 5.23
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    
    success: bool = Field(default=False, description="Operation success status")
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid image format",
                "details": "Only JPG and PNG formats are supported"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check"""
    
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    device: str = Field(description="Device being used (cuda/cpu)")
    
    model_config = {
        "protected_namespaces": (),  # Disable protected namespace warnings
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "cuda"
            }
        }
    }
