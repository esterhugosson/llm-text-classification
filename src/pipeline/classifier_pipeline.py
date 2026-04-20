# Classification pipeline - the core logic

from typing import Dict, Optional, Tuple, List
from src.data.models.data_models import Message, PredictionResult
from src.pipeline.filter import InteractionFilter


class ClassificationPipeline:
    """Handles the classification of messages"""
    
    def __init__(self, classifier, matcher, prompt_loader):
        """
        Args:
            classifier: LLM classifier (gpt_4o, claude, etc)
            matcher: GroundTruthMatcher instance
            prompt_loader: PromptLoader instance
        """
        self.classifier = classifier
        self.matcher = matcher
        self.prompt_loader = prompt_loader
    
    def classify_message(
        self,
        msg: Message,
        category: str,
        strategy: str,
        model_name: str
    ) -> Optional[PredictionResult]:
        """
        Classify a single message
        
        Args:
            msg: Message to classify
            category: Classification category
            strategy: Strategy (basic or few_shot)
            model_name: Name of the model
            
        Returns:
            PredictionResult or None if classification failed
        """
        try:
            # Get prompt
            prompt = self.prompt_loader.load_prompt(category, strategy)
            
            # Get true label from ground truth
            true_label = self.matcher.get_label(msg.thread_id, msg.message_id, category)
            print(f"True label for thread {msg.thread_id}, message {msg.message_id}, category {category}: {true_label}")
            if true_label is None:
                # ABORT MISSIONA
                return None # No ground truth for this category
            
            # Classify
            prediction = self.classifier.classify(prompt, msg.text)
            predicted_label = prediction.get("label")
            print(f"+++++")
            print(f"Predicted label: {predicted_label}, True label: {true_label}")
            print(f"+++++")
            
            # Check if valid
            if predicted_label is None or predicted_label == "ERROR":
                predicted_label = None
            
            assistant_name = self.matcher.get_assistant_name(msg.thread_id, msg.message_id)

            # Normalize both to string and lowercase (handle boolean values)
            pred_normalized = str(predicted_label).lower() if predicted_label is not None else None
            true_normalized = str(true_label).lower() if true_label is not None else None
            
            # Build result
            result = PredictionResult(
                thread_id=msg.thread_id,
                message_id=msg.message_id,
                assistant_name=assistant_name,
                text=msg.text[:150],
                category=category,
                strategy=strategy,
                model=model_name,
                role=msg.role,
                true_label=true_label,
                predicted_label=predicted_label,
                match = (
                    pred_normalized == true_normalized 
                    if pred_normalized is not None and true_normalized is not None else False
                )
            )
            
            return result
        
        except Exception as e:
            print(f"Error classifying msg {msg.message_id}: {e}")
            return None
    
    def classify_category(
        self,
        interactions: Dict[str, List[Dict]],
        category: str,
        strategy: str,
        model_name: str,
        role_filter: Optional[int] = None,
        message_limit: Optional[int] = None
    ) -> Tuple[List[PredictionResult], int]:
        """
        Classify all messages in a category with given strategy
        
        Args:
            interactions: Dict of thread_id -> messages
            category: Classification category
            strategy: Strategy (basic or few_shot)
            model_name: Name of the model
            role_filter: Role to filter by (0=teacher, 1=chatbot, None=all)
            message_limit: Max messages to classify
            
        Returns:
            (list of PredictionResult, count of classified messages)
        """
        
        msg_filter = InteractionFilter(required_role=role_filter, min_text_length=10)
        results = []
        category_count = 0
        
        # Loop through all messages
        for thread_id, messages_data in interactions.items():
            for msg_data in messages_data:
                msg_id = msg_data.get("id")
                text = msg_data.get("text", "").strip()
                msg_role = msg_data.get("role")
                
                # Create Message object
                msg = Message(
                    thread_id=thread_id,
                    message_id=msg_id,
                    text=text,
                    role=msg_role,
                )
                
                # Use filter to check if message should be classified
                if not msg_filter.allow(msg):
                    continue

                print(f"-------")
                print(f"-------")
                print(f"Classifying thread {thread_id}, message {msg_id} with role {msg_role}")
                print(f"-------")
                print(f"-------")
                
                # Classify
                result = self.classify_message(msg, category, strategy, model_name)

                print(f"Result for thread {thread_id}, message {msg_id}: {result}")
                
                if result is None:
                    continue
                
                results.append(result)
                category_count += 1
                
                # Check limit per category
                if message_limit and category_count >= message_limit:
                    break
            
            # Check limit per category
            if message_limit and category_count >= message_limit:
                break # Exit outer loop as well
        
        return results, category_count