"""
This module contains the FastAPI route handlers for the chat endpoint.
It will handle POST requests to /chat and manage the conversation flow.
"""
from fastapi import APIRouter, HTTPException, Request, Body
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import traceback
import json

from models import ChatRequest, ChatResponse, ChatMessage
from chat_llm import ChatLLM
from memory_store import MemoryStore
from utils import log_info, log_error

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Initialize router, memory store, and chat LLM
router = APIRouter()
memory_store = MemoryStore()
chat_llm = ChatLLM()

def build_system_prompt(quiz_answers: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a dynamic system prompt based on quiz answers and context.
    """
    logger.debug("="*50)
    logger.debug("BUILDING SYSTEM PROMPT")
    logger.debug("="*50)
    
    base_prompt = """You are a helpful and friendly supplement recommendation assistant. 
Your goal is to help customers find the right supplements based on their health goals and symptoms.
When appropriate, suggest relevant supplements from our catalog.
Be concise, friendly, and focus on being helpful."""

    if quiz_answers:
        logger.debug("Quiz answers provided:")
        logger.debug(json.dumps(quiz_answers, indent=2))
        # Add personalized context from quiz answers
        health_goals = quiz_answers.get("health_goals", [])
        symptoms = quiz_answers.get("symptoms", [])
        preferences = quiz_answers.get("preferences", {})
        
        context = []
        if health_goals:
            context.append(f"Health goals: {', '.join(health_goals)}")
        if symptoms:
            context.append(f"Current symptoms: {', '.join(symptoms)}")
        if preferences:
            dietary = preferences.get("dietary", [])
            if dietary:
                context.append(f"Dietary preferences: {', '.join(dietary)}")
        
        if context:
            base_prompt += "\n\nAdditional context:\n" + "\n".join(context)
            logger.debug("Added context to system prompt:")
            logger.debug("\n".join(context))
    
    logger.debug("Final system prompt:")
    logger.debug(base_prompt)
    logger.debug("="*50)
    return base_prompt

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest = Body(...)) -> ChatResponse:
    """
    Handle chat requests and generate responses.
    Uses Pydantic model for request validation.
    """
    logger.debug("="*50)
    logger.debug("CHAT REQUEST RECEIVED")
    logger.debug("="*50)
    
    try:
        # Log the request for debugging
        logger.debug("Request details:")
        logger.debug(json.dumps(request.dict(), indent=2, cls=DateTimeEncoder))

        # Extract fields from validated request
        message = request.message
        client_id = request.client_id
        session_id = request.session_id
        chat_history = request.chat_history
        quiz_answers = request.quiz_answers

        logger.debug("Extracted fields:")
        logger.debug(f"- Message: {message}")
        logger.debug(f"- Client ID: {client_id}")
        logger.debug(f"- Session ID: {session_id}")
        logger.debug(f"- Chat history length: {len(chat_history) if chat_history else 0}")
        logger.debug(f"- Quiz answers present: {'Yes' if quiz_answers else 'No'}")

        if not message:
            logger.error("Empty message received")
            raise HTTPException(status_code=422, detail="'message' field is required")

        # Store the user's message in memory
        user_message = ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.utcnow()
        )
        # Log the user message
        logger.debug("Created user message:")
        logger.debug(json.dumps(user_message.dict(), indent=2, cls=DateTimeEncoder))

        # Get or create session
        sid = session_id or memory_store.create_session(client_id)
        logger.debug(f"Using session ID: {sid}")

        # Store user message
        memory_store.add_message(sid, user_message.dict())
        logger.debug("Stored user message in memory")

        # Store quiz answers if provided
        if quiz_answers:
            logger.debug("Storing quiz answers:")
            logger.debug(json.dumps(quiz_answers, indent=2))
            memory_store.store_quiz_answers(sid, quiz_answers)
            logger.debug(f"Stored quiz answers for session {sid}")

        # Get chat history and quiz answers for context
        chat_hist = memory_store.get_messages(sid)
        stored_quiz_answers = memory_store.get_quiz_answers(sid)
        
        logger.debug("\nConversation context:")
        logger.debug(f"- Session ID: {sid}")
        logger.debug(f"- Messages in history: {len(chat_hist)}")
        logger.debug(f"- Has stored quiz answers: {'Yes' if stored_quiz_answers else 'No'}")
        if chat_hist:
            logger.debug("Chat history:")
            logger.debug(json.dumps([msg for msg in chat_hist], indent=2, cls=DateTimeEncoder))

        # Build dynamic system prompt
        logger.debug("\nBuilding system prompt...")
        system_prompt = build_system_prompt(stored_quiz_answers)

        # Generate response using LLM
        logger.debug("\nGenerating LLM response...")
        try:
            response = await chat_llm.generate_chat_response(
                message=message,
                chat_history=chat_hist,
                client_id=client_id,
                system_prompt=system_prompt
            )
            logger.debug("LLM response generated successfully:")
            logger.debug(json.dumps(response.dict(), indent=2, cls=DateTimeEncoder))
        except Exception as llm_exc:
            logger.error(f"LLM Error: {str(llm_exc)}", exc_info=True)
            response = ChatResponse(
                role="assistant",
                content="I'm sorry, I'm having trouble generating a response right now.",
                recommend=False
            )
            logger.debug("Created fallback response due to LLM error")

        # Create and store assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            timestamp=datetime.utcnow()
        )
        logger.debug("Created assistant message:")
        logger.debug(json.dumps(assistant_message.dict(), indent=2, cls=DateTimeEncoder))
        
        memory_store.add_message(sid, assistant_message.dict())
        logger.debug("Stored assistant message in memory")

        # Log successful response
        logger.info(f"Successfully processed chat request for session {sid}")
        logger.debug("="*50)
        return response

    except Exception as e:
        error_msg = f"Error processing chat request: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logger.debug("="*50)
        raise HTTPException(status_code=500, detail=error_msg) 
