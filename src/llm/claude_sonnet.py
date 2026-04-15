import anthropic
from dotenv import load_dotenv
from src.llm.base_classifier import BaseLLMClassifier
import os
from typing import Dict, List

load_dotenv()


class LLMClassifierClaudeSonnet(BaseLLMClassifier):
    def __init__(
            self, 
            model: str = "claude-sonnet-4-6",
            temperature: float = 0.0,
            max_tokens: int = 200,
        ):
        super().__init__(model, temperature, max_tokens)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Call ANTHROPIC API"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return completion.choices[0].message.content