import logging
import re
from datetime import datetime
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
from typing import Callable, Any, Optional
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def log_info(message: str) -> None:
    """
    Log an info message with timestamp.
    
    Args:
        message: The message to log
    """
    logger.info(message)

def log_error(message: str) -> None:
    """
    Log an error message with timestamp.
    
    Args:
        message: The error message to log
    """
    logger.error(message)

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
        log_error(f"Failed to parse float value: {value}")
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
        log_error(f"Failed to parse integer value: {value}")
        return default

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