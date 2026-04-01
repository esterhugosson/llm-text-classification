from src.llm.prompt_loader import PromptLoader
from src.llm.claude_sonnet import LLMClassifierClaudeSonnet

loader = PromptLoader()
classifier = LLMClassifierClaudeSonnet()

prompt = loader.load_prompt(
    category="interactional_move",
    strategy="basic"
)

text = "Låt oss utforska denna fråga tillsammans. Det är svårt att förutsäga resultatet av en fotbollsmatch eftersom det beror på många faktorer som lagens form, skador, taktik och dagsform. Vad tror du själv om chanserna för Öster? Finns det några specifika faktorer du tycker är viktiga för kvällens match?"

result = classifier.classify(prompt, text)

print(result)