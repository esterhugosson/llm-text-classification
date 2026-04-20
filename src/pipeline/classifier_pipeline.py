# Classification pipeline - the core logic

from typing import Dict, Optional, Tuple, List
from src.data.models.data_models import Message, PredictionResult
from src.pipeline.filter import InteractionFilter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ClassificationPipeline:
    """Handles the classification of messages"""
    
    # Initializes the classifier, the ground truth matcher and the loader for the prompts 
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
                logger.info("No ground truth to compare with, no classification done")
                return None # No ground truth for this category
            
            # LLM classify message based on prompt and the message's content.
            prediction = self.classifier.classify(prompt, msg.text)
            predicted_label = prediction.get("label")
            
            # Check if valid
            if predicted_label is None or predicted_label == "ERROR":
                logger.error("Error making prediction")
                predicted_label = None
            
            # Name of chatbot  
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
        
        message_filter = InteractionFilter(required_role=role_filter, min_text_length=10)
        results = []
        
        # Loop through all messages
        for thread_id, messages_data in interactions.items():
            for message_data in messages_data:
                # Extract message information to create Message object
                message_id = message_data.get("id")
                text = message_data.get("text", "").strip()
                message_role = message_data.get("role")
                
                # Create Message object
                message = Message(
                    thread_id=thread_id,
                    message_id=message_id,
                    text=text,
                    role=message_role,
                )
                
                # Use filter to check if message should be classified
                if not message_filter.allow(message):
                    continue
                
                # Classify
                result = self.classify_message(message, category, strategy, model_name)
                
                if result is None:
                    logger.info("No result from the classification process for message:", message_id)
                    continue
                
                results.append(result)
                
                # Check limit per category
                if message_limit and len(results) >= message_limit:
                    break
            
            # Check limit per category (I know, repetitive, but a must to break out of both loops)
            if message_limit and len(results) >= message_limit:
                break
        
        return results, len(results)