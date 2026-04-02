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

    def load_category(self, category: str) -> dict:
        """
        Return both prompts for one category
        """

        return {
            "basic": self.load_prompt(category, "basic"),
            "few_shot": self.load_prompt(category, "few_shot"),
        }

    def load_all_prompts(self) -> dict:
        categories = [
            "cps_behaviour",
            "response_substance",
            "response_stance",
            "interactional_move",
            "prompt_type",
            "is_followup",
        ]

        prompts = {}

        for cat in categories:
            prompts[cat] = self.load_category(cat)

        return prompts