from llm.prompt_loader import PromptLoader
from llm.claude_sonnet import LLMClassifierClaudeSonnet

loader = PromptLoader()
classifier = LLMClassifierClaudeSonnet()

prompt = loader.load_prompt(
    category="interactional_move",
    strategy="basic"
)

text = "Can you explain why this answer is incorrect?"

result = classifier.classify(prompt, text)

print(result)