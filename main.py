"""
Virtual Try-On API using FastAPI and IDM-VTON
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from typing import Optional

from config import settings
from schemas import TryOnResponse, ErrorResponse, HealthResponse
from utils import load_image, validate_image, encode_image_to_base64
from models import get_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Virtual Try-On API",
    description="AI-powered virtual try-on service using IDM-VTON",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    try:
        logger.info("Starting Virtual Try-On API...")
        logger.info(f"Device: {settings.device}")
        logger.info(f"Model: {settings.model_name}")
        
        # Pre-load the model
        model = get_model()
        logger.info("API ready!")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Virtual Try-On API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=model.model_loaded,
            device=model.device
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "device": "unknown",
                "error": str(e)
            }
        )


@app.post("/api/v1/tryon", response_model=TryOnResponse)
async def virtual_tryon(
    person_image: UploadFile = File(..., description="Image of the person"),
    garment_image: UploadFile = File(..., description="Image of the garment"),
    category: Optional[str] = Form("upper_body", description="Garment category"),
    num_inference_steps: Optional[int] = Form(30, description="Number of inference steps"),
    guidance_scale: Optional[float] = Form(2.0, description="Guidance scale")
):
    """
    Virtual try-on endpoint
    
    Upload a person image and a garment image to generate a try-on result.
    
    - **person_image**: Image of the person (JPG/PNG)
    - **garment_image**: Image of the garment (JPG/PNG)
    - **category**: Garment category (upper_body, lower_body, dresses)
    - **num_inference_steps**: Number of denoising steps (10-50, default: 30)
    - **guidance_scale**: Guidance scale (1.0-5.0, default: 2.0)
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        logger.info("Validating uploaded images...")
        validate_image(person_image)
        validate_image(garment_image)
        
        # Validate parameters
        if num_inference_steps < 10 or num_inference_steps > 50:
            raise HTTPException(
                status_code=400,
                detail="num_inference_steps must be between 10 and 50"
            )
        
        if guidance_scale < 1.0 or guidance_scale > 5.0:
            raise HTTPException(
                status_code=400,
                detail="guidance_scale must be between 1.0 and 5.0"
            )
        
        if category not in ["upper_body", "lower_body", "dresses"]:
            raise HTTPException(
                status_code=400,
                detail="category must be one of: upper_body, lower_body, dresses"
            )
        
        # Load images
        logger.info("Loading images...")
        person_img = await load_image(person_image)
        garment_img = await load_image(garment_image)
        
        # Get model and generate
        logger.info("Generating try-on result...")
        model = get_model()
        
        result_image = model.generate(
            person_image=person_img,
            garment_image=garment_img,
            category=category,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Encode result to base64
        result_base64 = encode_image_to_base64(result_image, format="PNG")
        
        processing_time = time.time() - start_time
        logger.info(f"Try-on completed in {processing_time:.2f} seconds")
        
        return TryOnResponse(
            success=True,
            message="Virtual try-on completed successfully",
            result_image=result_base64,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Try-on failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Try-on generation failed: {str(e)}"
        )


@app.get("/api/v1/models")
async def list_models():
    """List available models and configuration"""
    return {
        "current_model": settings.model_name,
        "device": settings.device,
        "supported_categories": ["upper_body", "lower_body", "dresses"],
        "max_upload_size": settings.max_upload_size,
        "allowed_formats": settings.get_allowed_extensions()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )
