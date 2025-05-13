"""
This module contains the FastAPI route handlers for the chat endpoint.
It will handle POST requests to /chat and manage the conversation flow.
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import traceback

from models import ChatRequest, ChatResponse, ChatMessage
from chat_llm import ChatLLM
from memory_store import MemoryStore
from utils import log_info, log_error

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize router, memory store, and chat LLM
router = APIRouter()
memory_store = MemoryStore()
chat_llm = ChatLLM()

def build_system_prompt(quiz_answers: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a dynamic system prompt based on quiz answers and context.
    
    Args:
        quiz_answers: Optional dictionary containing quiz answers
        
    Returns:
        str: Formatted system prompt
    """
    base_prompt = """You are a helpful and friendly supplement recommendation assistant. 
Your goal is to help customers find the right supplements based on their health goals and symptoms.
When appropriate, suggest relevant supplements from our catalog.
Be concise, friendly, and focus on being helpful."""

    if quiz_answers:
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
    
    return base_prompt

@router.post("/chat", response_model=ChatResponse)
async def chat(raw_request: Request) -> ChatResponse:
    """
    Handle chat requests and generate responses.
    Accepts any JSON body and adapts to the expected ChatRequest fields.
    """
    try:
        # Log the raw incoming body for debugging
        body_bytes = await raw_request.body()
        print("RAW REQUEST BODY:", body_bytes)
        data = await raw_request.json()
        print("PARSED JSON:", data)

        # Support both 'message' and 'messages' (array) payloads
        message = data.get("message")
        client_id = data.get("client_id")
        session_id = data.get("session_id")
        chat_history = data.get("chat_history")
        quiz_answers = data.get("quiz_answers")  # New field for quiz answers

        # If 'messages' array is present, use the last message's content
        if not message and "messages" in data and isinstance(data["messages"], list) and len(data["messages"]) > 0:
            last_msg = data["messages"][-1]
            message = last_msg.get("content")

        if not message:
            raise HTTPException(status_code=422, detail="'message' field is required (or 'messages' array with at least one message)")

        # Store the user's message in memory
        user_message = ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.utcnow()
        )

        # Get or create session
        sid = session_id or memory_store.create_session(client_id)
        logger.debug(f"Using session ID: {sid}")

        # Store user message
        memory_store.add_message(sid, user_message.dict())

        # Store quiz answers if provided
        if quiz_answers:
            memory_store.store_quiz_answers(sid, quiz_answers)
            print("\n" + "="*50)
            print("QUIZ ANSWERS & CONTEXT")
            print("="*50)
            print("Raw quiz answers:")
            for key, value in quiz_answers.items():
                print(f"- {key}: {value}")
            print("\nContext will be used to:")
            print("1. Understand user preferences")
            print("2. Guide product recommendations")
            print("3. Maintain conversation context")
            print("="*50 + "\n")

        # Get chat history and quiz answers for context
        chat_hist = memory_store.get_messages(sid)
        stored_quiz_answers = memory_store.get_quiz_answers(sid)
        
        # Log conversation statistics
        print("\n" + "="*50)
        print("CONVERSATION STATISTICS")
        print("="*50)
        print(f"Session ID: {sid}")
        print(f"Messages in history: {len(chat_hist)}")
        print(f"Has stored quiz answers: {'Yes' if stored_quiz_answers else 'No'}")
        print("="*50 + "\n")

        # Build dynamic system prompt
        system_prompt = build_system_prompt(stored_quiz_answers)

        # Generate response using LLM
        logger.debug("Generating response using LLM...")
        try:
            response = await chat_llm.generate_chat_response(
                message=message,
                chat_history=chat_hist,
                client_id=client_id,
                system_prompt=system_prompt
            )
        except Exception as llm_exc:
            print("LLM ERROR:", llm_exc)
            response = ChatResponse(
                role="assistant",
                content="I'm sorry, I'm having trouble generating a response right now.",
                recommend=False
            )

        # Create and store assistant message
        assistant_message = ChatMessage(
            role="assistant",
            content=response.content,
            timestamp=datetime.utcnow()
        )
        memory_store.add_message(sid, assistant_message.dict())

        # Log successful response
        logger.info(f"Generated response for session {sid}: {response.content}")

        # Log outgoing response for debugging
        print("OUTGOING RESPONSE:", {
            "role": response.role,
            "content": response.content,
            "recommend": response.recommend,
            "function_called": response.function_called,
            "function_name": response.function_name,
            "products_count": len(response.products) if response.products else 0
        })

        return response

    except Exception as e:
        error_msg = f"Error processing chat request: {str(e)}"
        print("EXCEPTION TRACE:")
        traceback.print_exc()
        log_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg) 