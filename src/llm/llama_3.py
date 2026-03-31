from ollama import chat
import json


class LLMClassifierLlama3:
    def __init__(self, model: str = "llama3:8b"):
        self.model = model

    def classify(self, prompt: str, text: str) -> dict:
        messages = [
            {
                "role": "system",
                "content": "You are a strict classifier that returns JSON only.",
            },
            {
                "role": "user",
                "content": prompt + "\n\n" + text,
            },
        ]

        response = chat(
            model=self.model,
            messages=messages,
        )

        content = response.message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": content}
