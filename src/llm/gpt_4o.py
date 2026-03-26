from openai import OpenAI
import json

class LLMClassifierGpt4o:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
    
    def classify(self, prompt: str, text: str) -> dict:
        """Classify with given prompt"""
        message = f"{prompt}\n\n{text}"
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            temperature=0,  # Deterministic
        )
        
        try:
            return json.loads(completion.choices[0].message.content)
        except json.JSONDecodeError:
            return {"error": completion.choices[0].message.content}