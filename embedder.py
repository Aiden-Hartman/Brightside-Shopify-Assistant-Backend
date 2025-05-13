import os
from typing import List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import time
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self):
        """Initialize the embedder with OpenAI API configuration."""
        logger.debug("Initializing Embedder...")
        self.api_key = os.getenv("OPENAI_API_KEY")
        logger.debug(f"OPENAI_API_KEY present: {'Yes' if self.api_key else 'No'}")
        
        if not self.api_key:
            error_msg = "OPENAI_API_KEY not found. Please add it to your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info("Initialized OpenAI embedder")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding vector using OpenAI API.
        
        Args:
            text: Input text to embed
            
        Returns:
            List[float]: 1536-dimensional embedding vector
            
        Raises:
            Exception: If embedding fails after retries
        """
        try:
            start_time = time.time()
            logger.debug(f"Generating embedding for text: {text[:100]}...")
            
            # Make API request
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            
            # Extract embedding vector
            embedding = response.data[0].embedding
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Generated embedding in {elapsed_time:.2f}s (vector size: {len(embedding)})")
            
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
        embeddings = []
        for i, text in enumerate(texts):
            logger.debug(f"Processing text {i+1}/{len(texts)}")
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings

    async def close(self):
        """Close the OpenAI client."""
        logger.debug("Closing OpenAI client...")
        await self.client.close() 