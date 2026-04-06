from src.experiments.config import GROUND_TRUTH_PATH

from src.data.loaders.ground_truth_loader import load_ground_truths

def main() :
    print(load_ground_truths(GROUND_TRUTH_PATH))