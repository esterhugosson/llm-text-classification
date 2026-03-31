from llm.llama_3 import LLMClassifierLlama3
from llm.prompt_loader import PromptLoader

loader = PromptLoader()
classifier = LLMClassifierLlama3()

prompt = loader.load_prompt(
    category="prompt_type",
    strategy="basic"
)

text = "make a picture from seed to plant and the circle of life for children"

result = classifier.classify(prompt, text)

print(result)