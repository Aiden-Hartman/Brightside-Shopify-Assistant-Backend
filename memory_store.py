"""
This module contains the MemoryStore class that manages chat history and session data.
"""
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime
import uuid

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

class MemoryStore:
    def __init__(self):
        """Initialize the memory store with empty dictionaries for messages and quiz answers."""
        logger.debug("="*50)
        logger.debug("INITIALIZING MEMORY STORE")
        logger.debug("="*50)
        
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.quiz_answers: Dict[str, Dict[str, Any]] = {}
        
        logger.debug("Initialized empty storage:")
        logger.debug(f"- Messages storage: {len(self.messages)} sessions")
        logger.debug(f"- Quiz answers storage: {len(self.quiz_answers)} sessions")
        logger.debug("="*50)

    def create_session(self, client_id: Optional[str] = None) -> str:
        """
        Create a new session and return its ID.
        """
        logger.debug("="*50)
        logger.debug("CREATING NEW SESSION")
        logger.debug("="*50)
        
        try:
            session_id = str(uuid.uuid4())
            logger.debug(f"Generated session ID: {session_id}")
            
            self.messages[session_id] = []
            self.quiz_answers[session_id] = {}
            
            logger.debug("Initialized session storage:")
            logger.debug(f"- Client ID: {client_id}")
            logger.debug(f"- Session ID: {session_id}")
            logger.debug(f"- Messages initialized: {len(self.messages[session_id])}")
            logger.debug(f"- Quiz answers initialized: {len(self.quiz_answers[session_id])}")
            
            logger.info(f"Created new session {session_id} for client {client_id}")
            logger.debug("="*50)
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    def add_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """
        Add a message to the session's chat history.
        """
        logger.debug("="*50)
        logger.debug("ADDING MESSAGE TO SESSION")
        logger.debug("="*50)
        
        try:
            logger.debug(f"Session ID: {session_id}")
            logger.debug("Message to add:")
            logger.debug(json.dumps(message, indent=2, cls=DateTimeEncoder))
            
            if session_id not in self.messages:
                logger.debug(f"Session {session_id} not found, creating new session")
                self.messages[session_id] = []
            
            self.messages[session_id].append(message)
            logger.debug(f"Message added. Session now has {len(self.messages[session_id])} messages")
            logger.debug("="*50)
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a session.
        """
        logger.debug("="*50)
        logger.debug("GETTING SESSION MESSAGES")
        logger.debug("="*50)
        
        try:
            logger.debug(f"Session ID: {session_id}")
            
            if session_id not in self.messages:
                logger.debug(f"Session {session_id} not found, returning empty list")
                return []
            
            messages = self.messages[session_id]
            logger.debug(f"Found {len(messages)} messages")
            logger.debug("Messages:")
            logger.debug(json.dumps(messages, indent=2, cls=DateTimeEncoder))
            
            logger.debug("="*50)
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    def store_quiz_answers(self, session_id: str, answers: Dict[str, Any]) -> None:
        """
        Store quiz answers for a session.
        """
        logger.debug("="*50)
        logger.debug("STORING QUIZ ANSWERS")
        logger.debug("="*50)
        
        try:
            logger.debug(f"Session ID: {session_id}")
            logger.debug("Quiz answers to store:")
            logger.debug(json.dumps(answers, indent=2, cls=DateTimeEncoder))
            
            if session_id not in self.quiz_answers:
                logger.debug(f"Session {session_id} not found, creating new session")
                self.quiz_answers[session_id] = {}
            
            self.quiz_answers[session_id] = answers
            logger.debug("Successfully stored quiz answers")
            logger.debug("="*50)
            
        except Exception as e:
            logger.error(f"Error storing quiz answers: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    def get_quiz_answers(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get quiz answers for a session.
        """
        logger.debug("="*50)
        logger.debug("GETTING QUIZ ANSWERS")
        logger.debug("="*50)
        
        try:
            logger.debug(f"Session ID: {session_id}")
            
            if session_id not in self.quiz_answers:
                logger.debug(f"Session {session_id} not found, returning None")
                return None
            
            answers = self.quiz_answers[session_id]
            logger.debug("Found quiz answers:")
            logger.debug(json.dumps(answers, indent=2, cls=DateTimeEncoder))
            
            logger.debug("="*50)
            return answers
            
        except Exception as e:
            logger.error(f"Error getting quiz answers: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise

    def clear_session(self, session_id: str) -> None:
        """
        Clear all data for a session.
        """
        logger.debug("="*50)
        logger.debug("CLEARING SESSION")
        logger.debug("="*50)
        
        try:
            logger.debug(f"Session ID: {session_id}")
            
            if session_id in self.messages:
                logger.debug(f"Clearing {len(self.messages[session_id])} messages")
                self.messages[session_id] = []
            
            if session_id in self.quiz_answers:
                logger.debug("Clearing quiz answers")
                self.quiz_answers[session_id] = {}
            
            logger.info(f"Cleared session {session_id}")
            logger.debug("="*50)
            
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}", exc_info=True)
            logger.debug("="*50)
            raise 
