# src/llm/base_classifier.py

from abc import ABC, abstractmethod
import json
from typing import Dict, Any, List


class BaseLLMClassifier(ABC):
    """Abstract base class for LLM classifiers"""
    
    DEFAULT_SYSTEM_PROMPT = "You are a strict classifier that returns JSON only."
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        """
        Initialize classifier with common parameters
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-6")
            temperature: Sampling temperature (0.0)
            max_tokens: Maximum tokens in response
            system_prompt: System message for the classifier
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
    
    def classify(self, prompt: str, text: str) -> Dict[str, Any]:
        """
        Classify text using the LLM
        
        Args:
            prompt: Classification prompt/instructions
            text: Text to classify
            
        Returns:
            Dictionary with classification result or error
        """
        # Build messages
        messages = self._build_messages(prompt, text)
        
        # Call the API (implemented by subclasses)
        response_text = self._call_api(messages)
        
        # Parse and return result
        return self._parse_response(response_text)
    
    def _build_messages(self, prompt: str, text: str) -> List[Dict[str, str]]:
        """
        Build message list in standard format
        
        Args:
            prompt: Classification prompt
            text: Text to classify
            
        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n{text}",
            },
        ]
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract JSON
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON or error dictionary
        """
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"error": f"Failed to parse JSON: {response_text}"}
    
    @abstractmethod
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM API (implemented by subclasses)
        
        Args:
            messages: Message list to send to API
            
        Returns:
            Response text from LLM
        """
        pass