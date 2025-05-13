"""
This module handles the OpenAI text generation logic for chat completions.
It will contain the core LLM interaction logic and response generation.
"""
import os
from typing import List, Dict, Tuple, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import logging
import time
from my_qdrant_utils import QdrantClient
from models import Product, ChatResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
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
        logger.debug("Initializing ChatLLM...")
        self.api_key = os.getenv("OPENAI_API_KEY")
        logger.debug(f"OPENAI_API_KEY present: {'Yes' if self.api_key else 'No'}")
        
        if not self.api_key:
            error_msg = "OPENAI_API_KEY not found. Please add it to your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.qdrant_client = QdrantClient()
        logger.info("Initialized OpenAI chat model with function calling capability")

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
        chat_history: List[Dict[str, str]],
        client_id: str = None,
        system_prompt: Optional[str] = None
    ) -> ChatResponse:
        """
        Generate a response using the OpenAI API with function calling capability.
        
        Args:
            message: Current user message
            chat_history: List of previous messages
            client_id: Optional client ID for tracking
            system_prompt: Optional custom system prompt
            
        Returns:
            ChatResponse: Response containing message and product information
            
        Raises:
            Exception: If generation fails after retries
        """
        try:
            start_time = time.time()
            logger.debug(f"Generating response for message: {message[:100]}...")
            
            # Format messages for OpenAI API
            messages = self._format_messages(message, chat_history, system_prompt)
            
            # Make API request with function calling
            logger.debug("Making request to OpenAI API...")
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                functions=[AVAILABLE_FUNCTIONS["query_supplements"]],
                function_call="auto",
                temperature=0.7,
                max_tokens=250
            )
            
            message = response.choices[0].message
            products = None
            function_called = False
            function_name = None
            
            if message.function_call:
                function_called = True
                function_name = message.function_call.name
                function_args = eval(message.function_call.arguments)
                
                # Execute the function and get products
                products = await self._handle_function_call(function_name, function_args)
                
                # Get final response with function result
                messages.append(message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(products)  # Convert products to string for the model
                })
                
                response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=250
                )
                generated_text = response.choices[0].message.content.strip()
            else:
                generated_text = message.content.strip()
            
            # Determine if we should recommend products
            should_recommend = bool(products) or any(keyword in generated_text.lower() 
                                                   for keyword in ["recommend", "suggest", "try", "consider", "look at"])
            
            # Log response details
            elapsed_time = time.time() - start_time
            print("\n" + "="*50)
            print("RESPONSE GENERATION")
            print("="*50)
            print(f"Time taken: {elapsed_time:.2f}s")
            print(f"Response length: {len(generated_text)} characters")
            print(f"Should recommend products: {should_recommend}")
            print(f"Function called: {function_called}")
            if function_called:
                print(f"Function name: {function_name}")
            if products:
                print(f"Number of products returned: {len(products)}")
            print("\nGenerated Response:")
            print("-"*30)
            print(generated_text)
            print("-"*30)
            print("="*50 + "\n")
            
            return ChatResponse(
                role="assistant",
                content=generated_text,
                recommend=should_recommend,
                products=products,
                function_called=function_called,
                function_name=function_name
            )
            
        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}", exc_info=True)
            raise

    async def close(self):
        """Close the OpenAI client."""
        logger.debug("Closing OpenAI client...")
        await self.client.close() 