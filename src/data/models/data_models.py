# Data structures for the project

from dataclasses import dataclass
from typing import Dict, Any, Optional


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
    true_label: str
    predicted_label: Optional[str]
    match: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "text": self.text,
            "category": self.category,
            "strategy": self.strategy,
            "model": self.model,
            "role": self.role,
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "match": self.match,
        }