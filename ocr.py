"""
OCR module for Vision Proxy.
Handles image extraction from messages and OCR API calls.
"""

import asyncio
from typing import Any
import httpx

from config import config


class OCRError(Exception):
    """Exception raised when OCR processing fails."""
    def __init__(self, message: str, image_index: int = -1):
        self.message = message
        self.image_index = image_index
        super().__init__(self.message)


def extract_images(messages: list[dict[str, Any]]) -> list[tuple[int, int, dict]]:
    """
    Extract all images from messages.
    
    Args:
        messages: List of OpenAI-format messages
        
    Returns:
        List of tuples: (message_index, content_index, image_content)
        where image_content is the {"type": "image_url", "image_url": {...}} dict
    """
    images = []
    
    for msg_idx, message in enumerate(messages):
        content = message.get("content")
        
        # Handle array-style content (multimodal messages)
        if isinstance(content, list):
            for content_idx, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "image_url":
                    images.append((msg_idx, content_idx, item))
    
    return images


async def call_ocr(image_content: dict, image_index: int) -> str:
    """
    Call OCR endpoint for a single image.
    
    Args:
        image_content: The image_url content dict from OpenAI format
        image_index: Index of the image (for error reporting)
        
    Returns:
        OCR result text
        
    Raises:
        OCRError: If OCR call fails
    """
    # Build request payload for OCR
    ocr_payload = {
        "model": config.OCR_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.OCR_PROMPT},
                    image_content
                ]
            }
        ],
        "max_tokens": 4096
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if configured
    if config.OCR_API_KEY:
        headers["Authorization"] = f"Bearer {config.OCR_API_KEY}"
    
    try:
        async with httpx.AsyncClient(timeout=config.OCR_TIMEOUT) as client:
            response = await client.post(
                config.OCR_ENDPOINT,
                json=ocr_payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise OCRError(
                    f"OCR API returned status {response.status_code}: {response.text}",
                    image_index
                )
            
            result = response.json()
            
            # Extract text from OpenAI-format response
            try:
                ocr_text = result["choices"][0]["message"]["content"]
                return ocr_text
            except (KeyError, IndexError) as e:
                raise OCRError(
                    f"Invalid OCR response format: {e}",
                    image_index
                )
                
    except httpx.TimeoutException:
        raise OCRError(f"OCR request timed out after {config.OCR_TIMEOUT}s", image_index)
    except httpx.RequestError as e:
        raise OCRError(f"OCR request failed: {e}", image_index)


async def process_images(images: list[tuple[int, int, dict]]) -> list[tuple[int, str]]:
    """
    Process all images through OCR.
    
    Args:
        images: List of (message_index, content_index, image_content) tuples
        
    Returns:
        List of (image_number, ocr_result) tuples, 1-indexed
        
    Raises:
        OCRError: If any OCR call fails
    """
    if not images:
        return []
    
    if config.OCR_PARALLEL:
        # Parallel processing
        tasks = [
            call_ocr(img[2], idx + 1)
            for idx, img in enumerate(images)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        ocr_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                if isinstance(result, OCRError):
                    raise result
                raise OCRError(f"OCR failed for image {idx + 1}: {result}", idx + 1)
            ocr_results.append((idx + 1, result))
        
        return ocr_results
    else:
        # Serial processing
        ocr_results = []
        for idx, (msg_idx, content_idx, image_content) in enumerate(images):
            result = await call_ocr(image_content, idx + 1)
            ocr_results.append((idx + 1, result))
        
        return ocr_results
