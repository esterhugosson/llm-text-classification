from llm.prompt_loader import PromptLoader
from llm.gpt_4o import LLMClassifierGpt4o

loader = PromptLoader()
classifier = LLMClassifierGpt4o()

prompt = loader.load_prompt(
    category="prompt_type",
    strategy="few_shot"
)

""" text = "Vilka är tänkbara efterträdare?" truth: request_example"""

text = "make a picture from seed to plant and the circle of life for children"

""" "prompt_type": "specific_request","""

result = classifier.classify(prompt, text)

print(result)
