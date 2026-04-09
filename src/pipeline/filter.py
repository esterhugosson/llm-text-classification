# Message filtering logic

from typing import Optional
from src.data.models.data_models import Message


class InteractionFilter:
    """Filter messages based on role and content criteria"""
    
    def __init__(self, required_role: Optional[int] = None, min_text_length: int = 10):
        """
        Args:
            required_role: 0=user, 1=chatbot, None=allow all
            min_text_length: Minimum text length to include
        """
        self.required_role = required_role
        self.min_text_length = min_text_length
    
    def allow(self, msg: Message) -> bool:
        """Check if message passes all filters"""
        # Check role
        if self.required_role is not None:
            if msg.role != self.required_role:
                return False
        
        # Check text length
        if len(msg.text.strip()) < self.min_text_length:
            return False
        
        # All checks passed
        return True
    
    def filter_messages(self, messages: list[Message]) -> list[Message]:
        """Filter a list of messages"""
        return [msg for msg in messages if self.allow(msg)]