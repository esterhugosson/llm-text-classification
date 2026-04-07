from src.experiments.config import GROUND_TRUTH_PATH

from src.data.loaders.ground_truth_loader import load_ground_truths
from src.pipeline.matcher import GroundTruthMatcher


exInteraction = {
    "thread_id": "thread_utJ1F4Ta8Ft9fIANC1fmHDx8",
    "message_id": 2701,
    "text": "Kan du hjälpa mig att skriva en social berättelse för en 11-åring som inte vill komma till skolan?",
    "category": "prompt_type",
    "strategy": "basic",
    "true_label": "specific_request",
    "predicted_label": "elaborated_request",
    "role": 0
  }


def main():

    truths = load_ground_truths(GROUND_TRUTH_PATH)
    matcher = GroundTruthMatcher(truths)

    true_label = matcher.get_label('thread_iyHh5IENSXUZYUCjcWYU3Efr', 2285, 'cps_behavior') # Ger labeln för kategorin

    get_truth = matcher.get_truth('thread_iyHh5IENSXUZYUCjcWYU3Efr', 2285) # Ger hela objektet

    print(get_truth)

