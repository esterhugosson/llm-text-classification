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
        system_prompt = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,  
            messages=filtered_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return response.content[0].text