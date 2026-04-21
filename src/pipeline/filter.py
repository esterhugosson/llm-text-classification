# Message filtering logic for roles regarding categories

from typing import Optional
from src.data.models.data_models import Message


class InteractionFilter:
    def __init__(self, required_role: Optional[int] = None):
        """
        Args:
            required_role: 0=user, 1=chatbot, None=allow all
        """
        self.required_role = required_role
    
    def allow(self, msg: Message) -> bool:
        # Check which role is required for the running category
        if self.required_role is not None:
            if msg.role != self.required_role:
                return False
        
        # Returns true if message has right role for the category (if none, which means both roles, then allowed)
        return True
    
    def filter_messages(self, messages: list[Message]) -> list[Message]:
        return [msg for msg in messages if self.allow(msg)]