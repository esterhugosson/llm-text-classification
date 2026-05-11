# Classification pipeline - the core logic

from typing import Dict, Optional, Tuple, List
from src.data.models.data_models import Message, PredictionResult
from src.pipeline.filter import InteractionFilter
from src.utils.logger import get_logger
from src.views.builder import ExperimentViewBuilder
import time

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
        self.view = ExperimentViewBuilder()

    def _get_context_messages(
        self,
        messages_data: List[Dict],
        current_idx: int,
        thread_id: str,
        max_previous: int = 3,
        max_next: int = 1
    ) -> List[Message]:
        """
        Get context messages around the current message:
        - Up to max_previous messages before current
        - The current message itself
        - Up to max_next messages after current

        Args:
            messages_data: Full list of messages in the thread
            current_idx: Index of the current message being classified
            thread_id: The thread ID
            max_previous: Maximum number of previous messages to include
            max_next: Maximum number of next messages to include

        Returns:
            List of Messages in chronological order
        """
        context_messages = []

        # Get previous messages (up to max_previous)
        start_idx = max(0, current_idx - max_previous)
        for i in range(start_idx, current_idx):
            msg_data = messages_data[i]
            message = Message(
                thread_id=thread_id,
                message_id=msg_data.get("id"),
                text=msg_data.get("text", "").strip(),
                role=msg_data.get("role"),
            )
            context_messages.append(message)

        # Get the current message
        current_data = messages_data[current_idx]
        current_message = Message(
            thread_id=thread_id,
            message_id=current_data.get("id"),
            text=current_data.get("text", "").strip(),
            role=current_data.get("role"),
        )
        context_messages.append(current_message)

        # Get next messages (up to max_next)
        end_idx = min(len(messages_data), current_idx + 1 + max_next)
        for i in range(current_idx + 1, end_idx):
            msg_data = messages_data[i]
            message = Message(
                thread_id=thread_id,
                message_id=msg_data.get("id"),
                text=msg_data.get("text", "").strip(),
                role=msg_data.get("role"),
            )
            context_messages.append(message)

        return context_messages

    def classify_message(
        self,
        msg: Message,
        category: str,
        strategy: str,
        model_name: str,
        context_messages: Optional[List[Message]] = None
    ) -> Optional[PredictionResult]:
        """
        Classify a single message.

        Args:
            msg: Message to classify
            category: Classification category
            strategy: Strategy (basic or few_shot)
            model_name: Name of the model
            context_messages: List of context messages (previous + current + next)

        Returns:
            PredictionResult or None if classification failed
        """

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                # Get prompt
                prompt = self.prompt_loader.load_prompt(category, strategy)

                # Replace [PREVIOUS_MESSAGE] placeholder with context
                if "[PREVIOUS_MESSAGE]" in prompt:
                    if context_messages:
                        context_lines = []
                        for ctx_msg in context_messages:
                            if ctx_msg.message_id == msg.message_id:
                                # Mark the current message being classified
                                context_lines.append(
                                    f">>> CLASSIFYING: [id={ctx_msg.message_id}] {ctx_msg.text} <<<"
                                )
                            else:
                                context_lines.append(
                                    f"[id={ctx_msg.message_id}] {ctx_msg.text}"
                                )
                        context_block = "\n".join(context_lines)
                    else:
                        context_block = "No context available"
                    prompt = prompt.replace("[PREVIOUS_MESSAGE]", context_block)
                
                # Get true label from ground truth
                true_label = self.matcher.get_label(msg.thread_id, msg.message_id, category)
                if true_label is None:
                    return None  # No ground truth for this category
                
                # LLM classify message based on prompt and the message's content
                prediction = self.classifier.classify(prompt)
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
                    text=msg.text,
                    category=category,
                    strategy=strategy,
                    model=model_name,
                    role=msg.role,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    match=(
                        pred_normalized == true_normalized 
                        if pred_normalized is not None and true_normalized is not None else False
                    )
                )
                
                return result
            
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in error_str or 
                    "rate_limit" in error_str or
                    "RateLimitError" in e.__class__.__name__
                )
                if is_rate_limit and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error classifying msg {msg.message_id}: {e}")
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
        Classify all messages in a category with given strategy.

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
        
        message_filter = InteractionFilter(required_role=role_filter)
        results = []
        
        for thread_id, messages_data in interactions.items():
            for idx, message_data in enumerate(messages_data):
                message_id = message_data.get("id")
                text = message_data.get("text", "").strip()
                message_role = message_data.get("role")
                
                message = Message(
                    thread_id=thread_id,
                    message_id=message_id,
                    text=text,
                    role=message_role,
                )
                
                if not message_filter.allow(message):
                    continue

                # Get context messages: max 3 previous + current + max 1 next
                context_messages = self._get_context_messages(messages_data, idx, thread_id, max_previous=3, max_next=1)

                result = self.classify_message(message, category, strategy, model_name, context_messages)
                
                if result is None:
                    continue
                
                results.append(result)

                self.view.print_classification_progress(result.message_id, result.predicted_label, result.match)
                
                if message_limit and len(results) >= message_limit:
                    break
            
            if message_limit and len(results) >= message_limit:
                break
        
        return results, len(results)