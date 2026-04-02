import os
import json
from dotenv import load_dotenv
import anthropic

load_dotenv()


class LLMClassifierClaudeSonnet:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    def classify(self, prompt: str, text: str) -> dict:

        system = "You are a strict classifier that returns JSON only."

        user_text = prompt + "\n\n" + text

        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=0,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}],
                }
            ],
        )

        content = response.content[0].text

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": content}
