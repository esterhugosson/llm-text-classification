from llm.prompt_loader import PromptLoader
from llm.gpt_4o import LLMClassifierGpt4o

loader = PromptLoader()
classifier = LLMClassifierGpt4o()

prompt = loader.load_prompt(
    category="prompt_type",
    strategy="basic"
)

""" text = "Vilka är tänkbara efterträdare?" truth: request_example"""

text = "Kan du hjälpa mig att skriva en social berättelse för en 11-åring som inte vill komma till skolan?" 

""" "prompt_type": "specific_request","""

result = classifier.classify(prompt, text)

print(result)
