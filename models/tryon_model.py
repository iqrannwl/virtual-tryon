"""
IDM-VTON Model Wrapper for Virtual Try-On
"""
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from typing import Optional, Tuple
import logging
from config import settings

logger = logging.getLogger(__name__)


class VirtualTryOnModel:
    """Wrapper for IDM-VTON virtual try-on model"""
    
    def __init__(self):
        """Initialize the model"""
        self.device = self._get_device()
        self.pipe = None
        self.model_loaded = False
        
    def _get_device(self) -> str:
        """Determine the device to use"""
        if settings.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif settings.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            logger.warning("CUDA/MPS not available, using CPU. This will be slow!")
            return "cpu"
    
    def load_model(self) -> None:
        """Load the virtual try-on model"""
        if self.model_loaded:
            logger.info("Model already loaded")
            return
        
        try:
            logger.info(f"Loading model {settings.model_name} on {self.device}...")
            
            # Load the IDM-VTON pipeline
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=settings.model_cache_dir,
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_vae_slicing()
            
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def preprocess_images(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        target_size: int = 768
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Preprocess images for the model
        
        Args:
            person_image: Image of the person
            garment_image: Image of the garment
            target_size: Target size for images
            
        Returns:
            Tuple of preprocessed (person_image, garment_image)
        """
        # Resize images while maintaining aspect ratio
        def resize_with_padding(img: Image.Image, size: int) -> Image.Image:
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            if width > height:
                new_width = size
                new_height = int(height * (size / width))
            else:
                new_height = size
                new_width = int(width * (size / height))
            
            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with padding
            new_img = Image.new("RGB", (size, size), (255, 255, 255))
            paste_x = (size - new_width) // 2
            paste_y = (size - new_height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            return new_img
        
        person_processed = resize_with_padding(person_image, target_size)
        garment_processed = resize_with_padding(garment_image, target_size)
        
        return person_processed, garment_processed
    
    def generate(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        category: str = "upper_body",
        num_inference_steps: int = 30,
        guidance_scale: float = 2.0,
    ) -> Image.Image:
        """
        Generate virtual try-on result
        
        Args:
            person_image: Image of the person
            garment_image: Image of the garment to try on
            category: Garment category (upper_body, lower_body, dresses)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            
        Returns:
            Generated try-on result image
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess images
            person_processed, garment_processed = self.preprocess_images(
                person_image, garment_image
            )
            
            logger.info(f"Generating try-on result with {num_inference_steps} steps...")
            
            # Generate the try-on result
            # Note: IDM-VTON uses a specific pipeline structure
            # This is a simplified version - you may need to adjust based on the actual model
            result = self.pipe(
                prompt=f"A photo of a person wearing the {category}",
                image=person_processed,
                mask_image=None,  # IDM-VTON handles masking internally
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            logger.info("Try-on generation completed!")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Try-on generation failed: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.model_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")


# Global model instance
_model_instance: Optional[VirtualTryOnModel] = None


def get_model() -> VirtualTryOnModel:
    """Get or create the global model instance"""
    global _model_instance
    
    if _model_instance is None:
        _model_instance = VirtualTryOnModel()
        _model_instance.load_model()
    
    return _model_instance
