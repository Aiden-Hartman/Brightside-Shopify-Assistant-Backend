"""
This module contains utility functions used across the application.
"""
import logging
import re
from datetime import datetime
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
from typing import Callable, Any, Optional, Dict
import unicodedata
import json

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

def log_info(message: str, data: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an informational message with optional data.
    """
    logger.debug("="*50)
    logger.debug("INFO LOG")
    logger.debug("="*50)
    
    try:
        logger.debug(f"Message: {message}")
        if data:
            logger.debug("Additional data:")
            logger.debug(json.dumps(data, indent=2))
        
        logger.info(message)
        logger.debug("="*50)
        
    except Exception as e:
        logger.error(f"Error in log_info: {str(e)}", exc_info=True)
        logger.debug("="*50)
        raise

def log_error(message: str, error: Exception, data: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error message with exception details and optional data.
    """
    logger.debug("="*50)
    logger.debug("ERROR LOG")
    logger.debug("="*50)
    
    try:
        logger.debug(f"Message: {message}")
        logger.debug(f"Error type: {type(error).__name__}")
        logger.debug(f"Error message: {str(error)}")
        
        if data:
            logger.debug("Additional data:")
            logger.debug(json.dumps(data, indent=2))
        
        logger.error(message, exc_info=True)
        logger.debug("="*50)
        
    except Exception as e:
        logger.error(f"Error in log_error: {str(e)}", exc_info=True)
        logger.debug("="*50)
        raise

def create_retry_decorator(
    max_attempts: int = 2,
    min_wait: int = 1,
    max_wait: int = 10
) -> Callable:
    """
    Create a retry decorator with customizable parameters.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries in seconds
        max_wait: Maximum wait time between retries in seconds
        
    Returns:
        Callable: Retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, min=min_wait, max=max_wait),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO),
        reraise=True
    )

def clean_description(text: str) -> str:
    """
    Clean product description text by:
    - Removing HTML tags
    - Converting Unicode characters to ASCII
    - Removing extra whitespace
    - Removing special characters
    
    Args:
        text: The text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Convert Unicode characters to ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()

def safe_parse_float(value: Any, default: float = 0.0) -> float:
    """
    Safely parse a value to float.
    
    Args:
        value: Value to parse
        default: Default value if parsing fails
        
    Returns:
        float: Parsed value or default
    """
    if value is None:
        return default
        
    try:
        return float(value)
    except (ValueError, TypeError):
        log_error(f"Failed to parse float value: {value}", None, None)
        return default

def safe_parse_int(value: Any, default: int = 0) -> int:
    """
    Safely parse a value to integer.
    
    Args:
        value: Value to parse
        default: Default value if parsing fails
        
    Returns:
        int: Parsed value or default
    """
    if value is None:
        return default
        
    try:
        return int(value)
    except (ValueError, TypeError):
        log_error(f"Failed to parse integer value: {value}", None, None)
        return default

def format_timestamp(timestamp: datetime) -> str:
    """
    Format a datetime object as an ISO 8601 string.
    """
    logger.debug("="*50)
    logger.debug("FORMATTING TIMESTAMP")
    logger.debug("="*50)
    
    try:
        logger.debug(f"Input timestamp: {timestamp}")
        formatted = timestamp.isoformat()
        logger.debug(f"Formatted timestamp: {formatted}")
        logger.debug("="*50)
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting timestamp: {str(e)}", exc_info=True)
        logger.debug("="*50)
        raise

def validate_json(data: Dict[str, Any]) -> bool:
    """
    Validate that a dictionary can be serialized to JSON.
    """
    logger.debug("="*50)
    logger.debug("VALIDATING JSON")
    logger.debug("="*50)
    
    try:
        logger.debug("Input data:")
        logger.debug(json.dumps(data, indent=2))
        
        # Try to serialize the data
        json.dumps(data)
        logger.debug("Data is valid JSON")
        logger.debug("="*50)
        return True
        
    except Exception as e:
        logger.error(f"Invalid JSON data: {str(e)}", exc_info=True)
        logger.debug("="*50)
        return False

def safe_json_loads(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse a JSON string, returning None if invalid.
    """
    logger.debug("="*50)
    logger.debug("SAFELY PARSING JSON")
    logger.debug("="*50)
    
    try:
        logger.debug(f"Input JSON string: {json_str}")
        result = json.loads(json_str)
        logger.debug("Successfully parsed JSON:")
        logger.debug(json.dumps(result, indent=2))
        logger.debug("="*50)
        return result
        
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}", exc_info=True)
        logger.debug("="*50)
        return None

# Example usage of retry decorator
@create_retry_decorator()
async def retry_async_function(func: Callable, *args, **kwargs) -> Any:
    """
    Wrap an async function with retry logic.
    
    Args:
        func: Async function to retry
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Any: Function result
    """
    return await func(*args, **kwargs) 
