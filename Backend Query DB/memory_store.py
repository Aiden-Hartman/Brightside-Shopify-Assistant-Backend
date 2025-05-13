"""
This module manages conversation memory and session tracking.
It will store chat history and maintain conversation context.
"""
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MemoryStore:
    def __init__(self):
        """Initialize the memory store with empty dictionaries."""
        logger.debug("Initializing MemoryStore...")
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._quiz_answers: Dict[str, Dict[str, Any]] = {}
        logger.info("MemoryStore initialized with empty storage")

    def create_session(self, client_id: Optional[str] = None) -> str:
        """
        Create a new session ID.
        
        Args:
            client_id: Optional client ID to associate with the session
            
        Returns:
            str: New session ID
        """
        session_id = str(uuid.uuid4())
        self._store[session_id] = []
        logger.debug(f"Created new session: {session_id} (client_id: {client_id})")
        return session_id

    def add_message(self, session_id: str, message: Dict[str, str]) -> None:
        """
        Add a message to the session history.
        
        Args:
            session_id: Session identifier
            message: Message to add (must contain 'role' and 'content')
            
        Raises:
            KeyError: If session_id doesn't exist
            ValueError: If message is invalid
        """
        try:
            if session_id not in self._store:
                logger.warning(f"Session {session_id} not found, creating new session")
                self._store[session_id] = []
            
            # Validate message format
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Message must be a dict with 'role' and 'content' keys")
            
            # Add message to history
            self._store[session_id].append(message)
            logger.debug(f"Added message to session {session_id}: {message['role']}: {message['content'][:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {str(e)}", exc_info=True)
            raise

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List[Dict[str, str]]: List of messages in the session
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        try:
            if session_id not in self._store:
                logger.warning(f"Session {session_id} not found, returning empty history")
                return []
            
            messages = self._store[session_id]
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages for session {session_id}: {str(e)}", exc_info=True)
            return []

    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages for a session.
        
        Args:
            session_id: Session identifier
            
        Raises:
            KeyError: If session_id doesn't exist
        """
        try:
            if session_id in self._store:
                self._store[session_id] = []
                logger.debug(f"Cleared all messages for session {session_id}")
            else:
                logger.warning(f"Attempted to clear non-existent session {session_id}")
                
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {str(e)}", exc_info=True)
            raise

    def get_all_sessions(self) -> List[str]:
        """
        Get a list of all active session IDs.
        
        Returns:
            List[str]: List of session IDs
        """
        try:
            sessions = list(self._store.keys())
            logger.debug(f"Retrieved {len(sessions)} active sessions")
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving session list: {str(e)}", exc_info=True)
            return []

    def store_quiz_answers(self, session_id: str, answers: Dict[str, Any]) -> None:
        """
        Store quiz answers for a session.
        
        Args:
            session_id: Session identifier
            answers: Dictionary containing quiz answers (e.g., budget, goals, preferences)
            
        Raises:
            ValueError: If answers is not a dictionary
        """
        try:
            if not isinstance(answers, dict):
                raise ValueError("Answers must be a dictionary")
            
            self._quiz_answers[session_id] = answers
            logger.debug(f"Stored quiz answers for session {session_id}: {answers}")
            
        except Exception as e:
            logger.error(f"Error storing quiz answers for session {session_id}: {str(e)}", exc_info=True)
            raise

    def get_quiz_answers(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get quiz answers for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[Dict[str, Any]]: Quiz answers if they exist, None otherwise
        """
        try:
            answers = self._quiz_answers.get(session_id)
            if answers:
                logger.debug(f"Retrieved quiz answers for session {session_id}")
            else:
                logger.debug(f"No quiz answers found for session {session_id}")
            return answers
            
        except Exception as e:
            logger.error(f"Error retrieving quiz answers for session {session_id}: {str(e)}", exc_info=True)
            return None 