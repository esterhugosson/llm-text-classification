# Ground truth matcher, to get the "true"(human annotated) label for chosen interaction more easy

class GroundTruthMatcher:

    def __init__(self, ground_truths):
        self.lookup = self._build_lookup(ground_truths)
        self.ground_truths = ground_truths


    # Method to quickly look up ground truth for a given thread_id and message_id, returns the whole dict for that message, which includes all categories and assistant name if available

    def _build_lookup(self, ground_truths) :
        
        # Build lookup: (thread_id, message_id) -> true_labels dict
        ground_truth_lookup = {}
        for thread_id, messages in ground_truths.items():
            for msg in messages:
                msg_id = msg.get("message_id")

                if msg_id is None:
                    continue

                key = (thread_id, msg_id)
                ground_truth_lookup[key] = msg

        return ground_truth_lookup
    
    def get_truth(self, thread_id, message_id):
        return self.lookup.get((thread_id, message_id))

    def get_label(self, thread_id, message_id, category):
        msg = self.get_truth(thread_id, message_id)
        if not msg:
            return None
        return msg.get(category)
    
    # Assistant meaning the chosen chatbot used during interaction
    def get_assistant_name(self, thread_id, message_id):
        msg = self.get_truth(thread_id, message_id)
        if not msg:
            return None
        return msg.get("assistant_name")


    # Just a temporary debug method
    def debug(self):
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print("THE LOOKUP LOOKS", self.lookup)
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")