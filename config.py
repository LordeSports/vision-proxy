"""
Configuration module for Vision Proxy.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # OCR Service Configuration
    OCR_ENDPOINT: str = os.getenv("OCR_ENDPOINT", "http://localhost:8080/v1/chat/completions")
    OCR_API_KEY: str = os.getenv("OCR_API_KEY", "")
    OCR_MODEL_NAME: str = os.getenv("OCR_MODEL_NAME", "gpt-4-vision")
    
    # OCR Processing Options
    OCR_PARALLEL: bool = os.getenv("OCR_PARALLEL", "true").lower() == "true"
    OCR_PROMPT: str = os.getenv("OCR_PROMPT", "请详细描述这张图片的内容")
    
    # Timeout Settings (seconds)
    OCR_TIMEOUT: int = int(os.getenv("OCR_TIMEOUT", "60"))
    UPSTREAM_TIMEOUT: int = int(os.getenv("UPSTREAM_TIMEOUT", "300"))
    
    # Proxy Server Settings
    PROXY_HOST: str = os.getenv("PROXY_HOST", "0.0.0.0")
    PROXY_PORT: int = int(os.getenv("PROXY_PORT", "8000"))


config = Config()
