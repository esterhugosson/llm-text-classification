# Data structures for the project

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union


@dataclass
class Message:
    """Represents a single message"""
    thread_id: str
    message_id: int
    text: str
    role: int  # 0 = teacher/user, 1 = chatbot/assistant
    

@dataclass
class GroundTruthLabel:
    """Ground truth labels for a message"""
    thread_id: str
    message_id: int
    labels: Dict[str, str]  # category -> label


@dataclass
class PredictionResult:
    """Single prediction result"""
    thread_id: str
    message_id: int
    text: str
    category: str
    strategy: str
    model: str
    role: int
    true_label: Union[str, bool]
    predicted_label: Optional[str]
    match: bool
    assistant_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "assistant_name": self.assistant_name,
            "text": self.text,
            "category": self.category,
            "strategy": self.strategy,
            "model": self.model,
            "role": self.role,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "match": self.match,
        }