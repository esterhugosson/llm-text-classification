# Ground truth lookup

class GroundTruthMatcher:
    def __init__(self, ground_truths):
        self.lookup = self._build_lookup(ground_truths)