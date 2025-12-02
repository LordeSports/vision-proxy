"""
Vision Proxy - Main Application

A proxy service that adds multimodal vision capability to text-only LLMs
by using an OCR model to describe images before forwarding to upstream.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
    
    Or run directly:
    python main.py
"""

import re
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response

from config import config
from ocr import extract_images, process_images, OCRError
from message_utils import format_ocr_results, rebuild_messages, has_images
from proxy import forward_chat_request, passthrough


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vision-proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Vision Proxy starting...")
    logger.info(f"OCR Endpoint: {config.OCR_ENDPOINT}")
    logger.info(f"OCR Model: {config.OCR_MODEL_NAME}")
    logger.info(f"OCR Parallel: {config.OCR_PARALLEL}")
    logger.info(f"OCR Timeout: {config.OCR_TIMEOUT}s")
    logger.info(f"Upstream Timeout: {config.UPSTREAM_TIMEOUT}s")
    yield
    logger.info("Vision Proxy shutting down...")


app = FastAPI(
    title="Vision Proxy",
    description="Proxy that adds vision capability to text-only LLMs via OCR",
    version="1.0.0",
    lifespan=lifespan
)


def parse_upstream_url(path: str) -> str | None:
    """
    Parse the upstream URL from the request path.
    
    Path format: /{upstream_url}
    Example: /http://172.17.0.1:7000/v1/chat/completions
           -> http://172.17.0.1:7000/v1/chat/completions
    
    Args:
        path: The request path
        
    Returns:
        The upstream URL or None if invalid
    """
    # Remove leading slash
    if path.startswith("/"):
        path = path[1:]
    
    # Validate it looks like a URL
    if not path.startswith(("http://", "https://")):
        return None
    
    return path


def is_chat_completions_path(url: str) -> bool:
    """
    Check if the URL path ends with v1/chat/completions.
    
    Args:
        url: The upstream URL
        
    Returns:
        True if path ends with v1/chat/completions
    """
    # Normalize and check
    return url.rstrip("/").endswith("v1/chat/completions")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_handler(request: Request, path: str):
    """
    Main proxy handler that routes requests based on path.
    
    For v1/chat/completions:
        - Extract images from messages
        - Send images to OCR endpoint
        - Append OCR results to messages
        - Remove images and forward to upstream
        
    For other paths:
        - Transparently forward without modification
    """
    # Parse upstream URL from path
    full_path = f"/{path}"
    if request.query_params:
        full_path += f"?{request.query_params}"
    
    upstream_url = parse_upstream_url(path)
    
    if not upstream_url:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid upstream URL in path. Expected format: /http://host/path or /https://host/path"
        )
    
    logger.info(f"Request: {request.method} -> {upstream_url}")
    
    # Get authorization header
    authorization = request.headers.get("Authorization")
    
    # Check if this is a chat completions request
    if is_chat_completions_path(upstream_url) and request.method == "POST":
        return await handle_chat_completions(request, upstream_url, authorization)
    else:
        # Transparent passthrough for other requests
        logger.info(f"Passthrough: {upstream_url}")
        try:
            return await passthrough(upstream_url, request)
        except Exception as e:
            logger.error(f"Passthrough error: {e}")
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


async def handle_chat_completions(
    request: Request,
    upstream_url: str,
    authorization: str | None
) -> Response:
    """
    Handle chat completions requests with image processing.
    
    Args:
        request: The incoming request
        upstream_url: The upstream API URL
        authorization: The Authorization header value
        
    Returns:
        Response from upstream
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
    
    messages = payload.get("messages", [])
    
    # Check if messages contain images
    if not has_images(messages):
        # No images, forward as-is
        logger.info("No images detected, forwarding as-is")
        try:
            return await forward_chat_request(upstream_url, payload, authorization)
        except Exception as e:
            logger.error(f"Upstream error: {e}")
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    
    # Extract images
    images = extract_images(messages)
    logger.info(f"Found {len(images)} image(s), processing with OCR...")
    
    # Process images through OCR
    try:
        ocr_results = await process_images(images)
    except OCRError as e:
        logger.error(f"OCR error: {e.message}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed for image {e.image_index}: {e.message}"
        )
    
    # Format OCR results
    ocr_text = format_ocr_results(ocr_results)
    logger.info(f"OCR completed, appending {len(ocr_results)} description(s)")
    
    # Rebuild messages without images, with OCR text appended
    new_messages = rebuild_messages(messages, images, ocr_text)
    
    # Update payload with new messages
    new_payload = payload.copy()
    new_payload["messages"] = new_messages
    
    # Forward to upstream
    try:
        return await forward_chat_request(upstream_url, new_payload, authorization)
    except Exception as e:
        logger.error(f"Upstream error: {e}")
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ocr_endpoint": config.OCR_ENDPOINT,
        "ocr_model": config.OCR_MODEL_NAME
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.PROXY_HOST,
        port=config.PROXY_PORT,
        reload=True
    )
