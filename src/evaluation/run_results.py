from pathlib import Path
from data.data_loader.data_loader import loadTruths
from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.prompt_loader import PromptLoader

def run_evaluation(sample_size: int = 50):
    loader = PromptLoader()
    classifier = LLMClassifierGpt4o(model="gpt-4o")
    truths = loadTruths()
    
    # Extract interactions
    all_interactions = [i for thread in truths.values() for i in thread]
    
    results = {}
    
    for category in loader.categories:
        prompts = {
            'zero_shot': loader.load_prompt(category, 'basic'),
            'few_shot': loader.load_prompt(category, 'few_shot')
        }
        
        for strategy, prompt in prompts.items():
            predictions = []
            ground_truths = []
            
            for interaction in all_interactions[:sample_size]:
                # Skip if label not applicable for this speaker
                if interaction.get(category) == 'none' or not interaction.get(category):
                    continue
                
                prediction = classifier.classify(prompt, interaction['text'])
                
                predictions.append(prediction)
                ground_truths.append(interaction[category])
            
            results[f"{category}_{strategy}"] = {
                'predictions': predictions,
                'ground_truths': ground_truths
            }
    
    return results