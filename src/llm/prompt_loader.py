from pathlib import Path

class PromptLoader:
    def __init__(self, prompts_dir: str = "src/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.categories = [
            'cps_behavior', 'response_substance', 'response_stance',
            'interactional_move', 'prompt_type', 'is_followup'
        ]
    
    def load_prompt(self, category: str, strategy: str) -> str:
        """strategy: 'basic' (zero-shot) or 'few_shot'"""
        path = self.prompts_dir / category / f'{strategy}.txt'
        return path.read_text(encoding='utf-8')
    
    def load_all_prompts(self) -> dict:
        """Return {category: {strategy: prompt_text}}"""
        prompts = {}
        for cat in self.categories:
            prompts[cat] = {
                'basic': self.load_prompt(cat, 'basic'),
                'few_shot': self.load_prompt(cat, 'few_shot')
            }
        return prompts