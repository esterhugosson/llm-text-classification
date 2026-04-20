from pathlib import Path


class PromptLoader:
    def __init__(self, prompts_dir: str = "src/prompts"):
        self.prompts_dir = Path(prompts_dir)

    def load_prompt(self, category: str, strategy: str) -> str:
        """
        category: interactional_move, cps_behavior, etc
        strategy: basic or few_shot
        """

        path = self.prompts_dir / category / f"{strategy}.txt"

        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")

        return path.read_text(encoding="utf-8")
