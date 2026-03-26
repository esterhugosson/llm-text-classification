from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()


class LLMClassifierGpt4o:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        content = completion.choices[0].message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": content}