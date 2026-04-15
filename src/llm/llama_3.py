from ollama import chat
from typing import Dict, List

from src.llm.base_classifier import BaseLLMClassifier


class LLMClassifierLlama3(BaseLLMClassifier):
    def __init__(
        self,
        model: str = "llama3:8b",
        temperature: float = 0.0,
        max_tokens: int = 200,
    ):
        super().__init__(model, temperature, max_tokens)
    
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama API and return response text"""
        response = chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        )
        return response.message.content