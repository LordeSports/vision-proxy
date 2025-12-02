"""
Message utilities for Vision Proxy.
Handles message transformation: removing images and appending OCR results.
"""

import copy
from typing import Any


def format_ocr_results(ocr_results: list[tuple[int, str]]) -> str:
    """
    Format OCR results into a readable string.
    
    Args:
        ocr_results: List of (image_number, ocr_text) tuples
        
    Returns:
        Formatted string like:
        图片1内容：(description1)
        图片2内容：(description2)
    """
    if not ocr_results:
        return ""
    
    lines = []
    for image_num, ocr_text in ocr_results:
        lines.append(f"图片{image_num}内容：({ocr_text})")
    
    return "\n".join(lines)


def rebuild_messages(
    messages: list[dict[str, Any]], 
    images: list[tuple[int, int, dict]],
    ocr_text: str
) -> list[dict[str, Any]]:
    """
    Rebuild messages by removing images and appending OCR descriptions.
    
    Args:
        messages: Original messages list
        images: List of (message_index, content_index, image_content) tuples
        ocr_text: Formatted OCR results to append
        
    Returns:
        New messages list with images removed and OCR text appended
    """
    # Deep copy to avoid modifying original
    new_messages = copy.deepcopy(messages)
    
    # Build a set of (msg_idx, content_idx) to remove
    images_to_remove = {(msg_idx, content_idx) for msg_idx, content_idx, _ in images}
    
    # Process each message
    for msg_idx, message in enumerate(new_messages):
        content = message.get("content")
        
        if isinstance(content, list):
            # Filter out image_url items that are in our removal set
            new_content = []
            for content_idx, item in enumerate(content):
                if (msg_idx, content_idx) not in images_to_remove:
                    new_content.append(item)
            
            # If only text items remain, simplify the content
            if len(new_content) == 0:
                message["content"] = ""
            elif len(new_content) == 1 and new_content[0].get("type") == "text":
                message["content"] = new_content[0].get("text", "")
            else:
                message["content"] = new_content
    
    # Append OCR text to the last user message
    if ocr_text:
        # Find the last user message
        last_user_idx = -1
        for idx in range(len(new_messages) - 1, -1, -1):
            if new_messages[idx].get("role") == "user":
                last_user_idx = idx
                break
        
        if last_user_idx >= 0:
            current_content = new_messages[last_user_idx].get("content", "")
            
            if isinstance(current_content, str):
                # String content - append directly
                if current_content:
                    new_messages[last_user_idx]["content"] = f"{current_content}\n\n{ocr_text}"
                else:
                    new_messages[last_user_idx]["content"] = ocr_text
            elif isinstance(current_content, list):
                # Array content - append as text item
                current_content.append({"type": "text", "text": f"\n\n{ocr_text}"})
        else:
            # No user message found, append as new user message
            new_messages.append({"role": "user", "content": ocr_text})
    
    return new_messages


def has_images(messages: list[dict[str, Any]]) -> bool:
    """
    Check if messages contain any images.
    
    Args:
        messages: List of OpenAI-format messages
        
    Returns:
        True if any message contains image_url content
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False
