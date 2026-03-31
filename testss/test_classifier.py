from llm.prompt_loader import PromptLoader
from llm.gpt_4o import LLMClassifierGpt4o

loader = PromptLoader()
classifier = LLMClassifierGpt4o()

prompt = loader.load_prompt(
    category="interactional_move",
    strategy="basic"
)

text = "Utan specifika nyheter eller en händelse för att precisera svaret, kan det finnas flera allmänna anledningar till varför en partiledare för ett politiskt parti som Liberalerna i Sverige kan välja att avgå. Dessa kan inkludera:\n\n1. **Personliga Skäl**: Partiledaren kan ha hälsoproblem, behöva mer tid för familjen, eller vara trött på den intensiva arbetsbelastningen.\n\n2. **Politisk Press**: Om partiet har presterat dåligt i val eller opinionsmätningar, kan det finnas press från partimedlemmar eller väljarbasen att göra förändringar i ledarskapet.\n\n3. **Inre Partikonflikter**: Det kan finnas oenigheter inom partiet om dess riktning, politik eller strategi, vilket kan leda till en avgång.\n\n4. **Strategiskt Drag**: Ibland kan en ledare välja att avgå som en del av en strategi för att ge en ny ledare chansen att bygga en ny image eller införa nya idéer.\n\n5. **Skandaler eller Kontroverser**: Eventuella skandaler eller kontroverser som involverar ledaren personligen kan också leda till avgång för att skydda partiets rykte.\n\nFör den mest exakta och uppdaterade informationen rekommenderas att kolla aktuella nyheter från pålitliga källor."


result = classifier.classify(prompt, text)

print(result)
