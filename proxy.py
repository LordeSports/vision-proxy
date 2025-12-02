"""
Proxy module for Vision Proxy.
Handles request forwarding and response streaming.
"""

from typing import Any, AsyncGenerator
import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse, Response

from config import config


async def forward_chat_request(
    upstream_url: str,
    payload: dict[str, Any],
    authorization: str | None
) -> Response | StreamingResponse:
    """
    Forward a chat completion request to upstream.
    
    Args:
        upstream_url: The upstream API URL
        payload: The request payload (modified messages)
        authorization: The original Authorization header value
        
    Returns:
        Response or StreamingResponse for streaming
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    if authorization:
        headers["Authorization"] = authorization
    
    # Check if streaming is requested
    is_streaming = payload.get("stream", False)
    
    if is_streaming:
        return await stream_response(upstream_url, payload, headers)
    else:
        return await non_stream_response(upstream_url, payload, headers)


async def non_stream_response(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str]
) -> Response:
    """
    Forward request and return non-streaming response.
    """
    async with httpx.AsyncClient(timeout=config.UPSTREAM_TIMEOUT) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.headers.get("content-type")
        )


async def stream_response(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str]
) -> StreamingResponse:
    """
    Forward request and stream response back.
    """
    async def generate() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=config.UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


async def passthrough(
    upstream_url: str,
    request: Request
) -> Response | StreamingResponse:
    """
    Transparently forward any request to upstream without modification.
    
    Args:
        upstream_url: The upstream URL to forward to
        request: The original FastAPI request
        
    Returns:
        Response from upstream
    """
    # Get request body
    body = await request.body()
    
    # Build headers, excluding host
    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ("host", "content-length"):
            headers[key] = value
    
    method = request.method
    
    async with httpx.AsyncClient(timeout=config.UPSTREAM_TIMEOUT) as client:
        response = await client.request(
            method=method,
            url=upstream_url,
            content=body,
            headers=headers,
            params=request.query_params
        )
        
        # Check if response is streaming
        content_type = response.headers.get("content-type", "")
        
        if "text/event-stream" in content_type:
            # Re-request with streaming
            async def generate() -> AsyncGenerator[bytes, None]:
                async with httpx.AsyncClient(timeout=config.UPSTREAM_TIMEOUT) as stream_client:
                    async with stream_client.stream(
                        method=method,
                        url=upstream_url,
                        content=body,
                        headers=headers,
                        params=request.query_params
                    ) as stream_response:
                        async for chunk in stream_response.aiter_bytes():
                            yield chunk
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        
        # Return normal response
        # Filter out hop-by-hop headers
        response_headers = {}
        hop_by_hop = {"connection", "keep-alive", "transfer-encoding", "te", "trailers", "upgrade"}
        for key, value in response.headers.items():
            if key.lower() not in hop_by_hop:
                response_headers[key] = value
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type")
        )
