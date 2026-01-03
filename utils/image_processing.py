"""
Image processing utilities
"""
from PIL import Image
import base64
import io
from typing import Union, Tuple
from fastapi import UploadFile, HTTPException
from config import settings


async def load_image(file: UploadFile) -> Image.Image:
    """
    Load an image from uploaded file
    
    Args:
        file: Uploaded file from FastAPI
        
    Returns:
        PIL Image object
        
    Raises:
        HTTPException: If image cannot be loaded
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to load image: {str(e)}"
        )


def validate_image(file: UploadFile) -> None:
    """
    Validate uploaded image file
    
    Args:
        file: Uploaded file from FastAPI
        
    Raises:
        HTTPException: If validation fails
    """
    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = file.filename.split(".")[-1].lower()
    allowed_exts = settings.get_allowed_extensions()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension. Allowed: {', '.join(allowed_exts)}"
        )
    
    # Check content type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Must be JPEG or PNG"
        )


def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG)
        
    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    img_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(img_data))
    return image.convert("RGB")


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]] = None
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        size: Target size (int for max dimension, or (width, height) tuple)
        
    Returns:
        Resized PIL Image object
    """
    if size is None:
        size = settings.default_image_size
    
    if isinstance(size, int):
        # Resize to max dimension while maintaining aspect ratio
        width, height = image.size
        if width > height:
            new_width = size
            new_height = int(height * (size / width))
        else:
            new_height = size
            new_width = int(width * (size / height))
        size = (new_width, new_height)
    
    return image.resize(size, Image.Resampling.LANCZOS)


def save_image(image: Image.Image, path: str) -> None:
    """
    Save PIL Image to file
    
    Args:
        image: PIL Image object
        path: File path to save to
    """
    image.save(path, quality=95)
