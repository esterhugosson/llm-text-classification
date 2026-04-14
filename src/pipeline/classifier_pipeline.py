# Classification pipeline - the core logic

import json
from typing import Dict, Any, Optional, Tuple, List
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
            if true_label is None:
                return None  # No ground truth for this category
            
            # Classify
            prediction = self.classifier.classify(prompt, msg.text)
            predicted_label = prediction.get("label")
            
            # Check if valid
            if predicted_label is None or predicted_label == "ERROR":
                predicted_label = None
            
            # Build result
            result = PredictionResult(
                thread_id=msg.thread_id,
                message_id=msg.message_id,
                text=msg.text[:150],
                category=category,
                strategy=strategy,
                model=model_name,
                role=msg.role,
                true_label=true_label,
                predicted_label=predicted_label,
                match=predicted_label == true_label if predicted_label else False,
            )
            
            return result
        
        except Exception as e:
            print(f"Error classifying msg {msg.message_id}: {e}")
            return None
    
    def classify_batch(
        self,
        messages: list[Message],
        category: str,
        strategy: str,
        model_name: str
    ) -> list[PredictionResult]:
        """
        Classify a batch of messages
        
        Returns:
            List of PredictionResult objects
        """
        results = []
        for msg in messages:
            result = self.classify_message(msg, category, strategy, model_name)
            if result is not None:
                results.append(result)
        
        return results
    
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
                
                # Use filter to check if message passes
                if not msg_filter.allow(msg):
                    continue
                
                # Classify
                result = self.classify_message(msg, category, strategy, model_name)
                
                if result is None:
                    continue
                
                results.append(result)
                category_count += 1
                
                # Check limit per category
                if message_limit and category_count >= message_limit:
                    break
        
        return results, category_count