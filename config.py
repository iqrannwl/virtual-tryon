"""
Configuration management for Virtual Try-On API
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Model Configuration
    model_name: str = "yisol/IDM-VTON"
    device: str = "cuda"  # 'cuda' or 'cpu'
    model_cache_dir: str = "./model_cache"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_upload_size: int = 10485760  # 10MB
    allowed_extensions: str = "jpg,jpeg,png"  # Will be converted to list
    
    # Processing Configuration
    default_image_size: int = 768
    enable_safety_checker: bool = False
    
    # CORS Configuration
    cors_origins: str = "*"  # Will be converted to list
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "protected_namespaces": ()  # Disable protected namespace warnings
    }
    
    def get_allowed_extensions(self) -> List[str]:
        """Get allowed extensions as a list"""
        if isinstance(self.allowed_extensions, str):
            return [ext.strip() for ext in self.allowed_extensions.split(',') if ext.strip()]
        return self.allowed_extensions
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as a list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(',') if origin.strip()]
        return self.cors_origins


# Global settings instance
settings = Settings()


# Create necessary directories
os.makedirs(settings.model_cache_dir, exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
