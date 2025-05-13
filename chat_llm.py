"""
This module handles the OpenAI text generation logic for chat completions.
It will contain the core LLM interaction logic and response generation.
"""
import os
from typing import List, Dict, Tuple, Optional, Any
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import logging
import time
from my_qdrant_utils import QdrantClient
from models import Product, ChatResponse, ChatMessage
from utils import log_info, log_error
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = """You are a helpful and friendly supplement recommendation assistant. 
Your goal is to help customers find the right supplements based on their health goals and symptoms.
When appropriate, suggest relevant supplements from our catalog.
Be concise, friendly, and focus on being helpful."""

# Define available functions for the model
AVAILABLE_FUNCTIONS = {
    "query_supplements": {
        "name": "query_supplements",
        "description": "Query the supplement database for relevant products based on a health-related question",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's health-related question or concern"
                }
            },
            "required": ["question"]
        }
    }
}

class ChatLLM:
    def __init__(self):
        """Initialize the chat LLM with OpenAI API configuration."""
        logger.debug("="*50)
        logger.debug("INITIALIZING CHAT LLM")
        logger.debug("="*50)
        
        try:
            # Initialize OpenAI client
            logger.debug("Initializing OpenAI client...")
            self.api_key = os.getenv("OPENAI_API_KEY")
            logger.debug(f"OPENAI_API_KEY present: {'Yes' if self.api_key else 'No'}")
            
            if not self.api_key:
                error_msg = "OPENAI_API_KEY not found. Please add it to your .env file."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.debug("Successfully initialized OpenAI client")
            
            # Initialize Qdrant client
            logger.debug("Initializing Qdrant client...")
            self.qdrant_client = QdrantClient()
            logger.debug("Successfully initialized Qdrant client")
            
            logger.info("Initialized OpenAI chat model with function calling capability")
        except Exception as e:
            logger.error(f"Error initializing ChatLLM: {str(e)}", exc_info=True)
            raise
        
        logger.debug("="*50)

    def _format_messages(self, message: str, chat_history: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format messages for OpenAI API.
        
        Args:
            message: Current user message
            chat_history: List of previous messages
            system_prompt: Optional custom system prompt
            
        Returns:
            List[Dict[str, str]]: Formatted messages for OpenAI API
        """
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": system_prompt or DEFAULT_SYSTEM_PROMPT
        })
        
        # Add chat history
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        return messages

    async def _handle_function_call(self, function_name: str, function_args: Dict) -> List[Product]:
        """
        Handle function calls from the model.
        
        Args:
            function_name: Name of the function to call
            function_args: Arguments for the function
            
        Returns:
            List[Product]: List of products returned by the function
        """
        if function_name == "query_supplements":
            question = function_args.get("question")
            if not question:
                logger.error("No question provided for supplement query")
                return []
            
            try:
                # Get embedding for the question
                from embedder import Embedder
                embedder = Embedder()
                query_vector = await embedder.embed_text(question)
                
                # Query Qdrant
                products = await self.qdrant_client.query_qdrant(
                    query_vector=query_vector,
                    limit=3
                )
                
                return products
                
            except Exception as e:
                logger.error(f"Error in query_supplements: {str(e)}", exc_info=True)
                return []
        
        logger.error(f"Unknown function {function_name}")
        return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_chat_response(
        self,
        message: str,
        chat_history: List[Dict[str, Any]],
        client_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> ChatResponse:
        """
        Generate a chat response using the language model.
        """
        logger.debug("="*50)
        logger.debug("GENERATING CHAT RESPONSE")
        logger.debug("="*50)
        
        try:
            # Log input parameters
            logger.debug("Input parameters:")
            logger.debug(f"- Message: {message}")
            logger.debug(f"- Client ID: {client_id}")
            logger.debug(f"- System prompt present: {'Yes' if system_prompt else 'No'}")
            logger.debug(f"- Chat history length: {len(chat_history)}")
            if chat_history:
                logger.debug("Chat history:")
                logger.debug(json.dumps(chat_history, indent=2))

            # Prepare messages for the chat completion
            logger.debug("\nPreparing messages for chat completion...")
            messages = self._format_messages(message, chat_history, system_prompt)
            
            # Generate embedding for the message
            logger.debug("\nGenerating message embedding...")
            try:
                embedding_response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=message
                )
                query_vector = embedding_response.data[0].embedding
                logger.debug(f"Successfully generated embedding of length {len(query_vector)}")
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
                raise

            # Query Qdrant for relevant products
            logger.debug("\nQuerying Qdrant for relevant products...")
            try:
                products = await self.qdrant_client.query_qdrant(
                    query_vector=query_vector,
                    limit=3,
                    client_id=client_id
                )
                logger.debug(f"Found {len(products)} relevant products")
                if products:
                    logger.debug("Product details:")
                    logger.debug(json.dumps([p.dict() for p in products], indent=2))
            except Exception as e:
                logger.error(f"Error querying Qdrant: {str(e)}", exc_info=True)
                products = []

            # Generate chat completion
            logger.debug("\nGenerating chat completion...")
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                response_content = completion.choices[0].message.content
                logger.debug("Successfully generated chat completion:")
                logger.debug(response_content)
            except Exception as e:
                logger.error(f"Error generating chat completion: {str(e)}", exc_info=True)
                raise

            # Create response
            logger.debug("\nCreating ChatResponse object...")
            response = ChatResponse(
                role="assistant",
                content=response_content,
                recommend=len(products) > 0,
                products=products
            )
            logger.debug("Created response:")
            logger.debug(json.dumps(response.dict(), indent=2))

            logger.info("Successfully generated chat response")
            logger.debug("="*50)
            return response

        except Exception as e:
            logger.error(f"Error in generate_chat_response: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    async def close(self):
        """Close the OpenAI client."""
        logger.debug("="*50)
        logger.debug("CLOSING CHAT LLM")
        logger.debug("="*50)
        try:
            logger.debug("Closing OpenAI client...")
            await self.client.close()
            logger.debug("Successfully closed OpenAI client")
            
            # Close Qdrant client
            await self.qdrant_client.close()
            logger.debug("Successfully closed Qdrant client")
        except Exception as e:
            logger.error(f"Error closing ChatLLM: {str(e)}", exc_info=True)
            raise
        logger.debug("="*50) 
