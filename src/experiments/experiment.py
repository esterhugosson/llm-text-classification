# Main experiment orchestration

import sys
import traceback
from datetime import datetime
from typing import Optional, List

from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.claude_sonnet import LLMClassifierClaudeSonnet
from src.llm.llama_3 import LLMClassifierLlama3

from src.experiments.config import MODELS, CATEGORIES, STRATEGIES
from src.experiments.stats import ExperimentStats

from src.data.loaders.ground_truth_loader import load_ground_truths
from src.data.loaders.interaction_loader import load_interactions
from src.data.models.data_models import Message

from src.pipeline.matcher import GroundTruthMatcher
from src.pipeline.filter import InteractionFilter
from src.pipeline.classifier_pipeline import ClassificationPipeline
from src.pipeline.result_builder import ResultBuilder

from src.llm.prompt_loader import PromptLoader


class Experiment:
    """Main experiment runner"""
    
    def __init__(
        self,
        interactions_path: str = "data/dataset/interactions.json",
        ground_truth_path: str = "data/process_data/processed_ground_truths.json",
        message_limit: Optional[int] = None,
    ):
        """Initialize experiment with data paths"""
        self.interactions_path = interactions_path
        self.ground_truth_path = ground_truth_path
        self.message_limit = message_limit
        
        self.stats = ExperimentStats()
        self.result_builder = ResultBuilder()
        
        # Will be loaded when running
        self.interactions = None
        self.ground_truths = None
        self.matcher = None
        self.prompt_loader = PromptLoader()
    
    def _load_data(self):
        """Load interactions and ground truth"""
        print("Loading data...")
        
        self.interactions = load_interactions(self.interactions_path)
        self.ground_truths = load_ground_truths(self.ground_truth_path)
        self.matcher = GroundTruthMatcher(self.ground_truths)
        
        # Count messages
        total_messages = sum(len(msgs) for msgs in self.interactions.values())
        
        print(f"   ✓ Loaded {len(self.interactions)} threads")
        print(f"   ✓ Loaded {total_messages} messages")
        print(f"   ✓ Loaded ground truth labels\n")
    
    def _get_classifier(self, model_name: str):
        """Get LLM classifier instance"""
        if model_name == "gpt4o":
            return LLMClassifierGpt4o()
        elif model_name == "claude":
            return LLMClassifierClaudeSonnet()
        elif model_name == "llama3":
            return LLMClassifierLlama3()
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def run(
        self,
        models: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        """
        Run the full experiment
        
        Args:
            models: List of models to use (default: all)
            categories: List of categories to classify (default: all)
            
        Returns:
            Path to saved results file
        """
        
        # Use all if not specified
        if models is None:
            models = list(MODELS.keys())
        if categories is None:
            categories = list(CATEGORIES.keys())
        
        # Load data
        self._load_data()
        
        # Print header
        print(f"\n{'='*70}")
        print(f"  STARTING EXPERIMENT")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Models: {', '.join(models)}")
        print(f"  Categories: {', '.join(categories)}")
        print(f"{'='*70}\n")
        
        # Loop: model → category → strategy → messages
        for model_idx, model_name in enumerate(models, 1):
            print(f"\n{'─'*70}")
            print(f"  [{model_idx}/{len(models)}] Model: {model_name}")
            print(f"{'─'*70}")
            
            try:
                classifier = self._get_classifier(model_name)
            except ValueError as e:
                print(f"  {e}")
                continue
            
            # Create pipeline for this model
            pipeline = ClassificationPipeline(
                classifier=classifier,
                matcher=self.matcher,
                prompt_loader=self.prompt_loader,
            )
            
            for cat_idx, (category, role_filter) in enumerate(CATEGORIES.items(), 1):
                # Skip if not in requested categories
                if category not in categories:
                    continue
                
                role_label = {0: "teacher", 1: "chatbot", None: "all"}.get(
                    role_filter, role_filter
                )
                print(f"\n  [{cat_idx}/{len(CATEGORIES)}] {category} (role={role_label})")
                
                for strat_idx, strategy in enumerate(STRATEGIES, 1):
                    print(f"    [{strat_idx}/{len(STRATEGIES)}] {strategy}")
                    
                    try:
                        # Create filter for this category
                        msg_filter = InteractionFilter(
                            required_role=role_filter,
                            min_text_length=10
                        )
                        
                        category_count = 0
                        
                        # Loop through all messages
                        for thread_id, messages_data in self.interactions.items():
                            for msg_data in messages_data:
                                msg_id = msg_data.get("message_id")
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
                                    self.stats.increment_skip()
                                    continue
                                
                                # Classify
                                result = pipeline.classify_message(
                                    msg=msg,
                                    category=category,
                                    strategy=strategy,
                                    model_name=model_name,
                                )
                                
                                if result is None:
                                    self.stats.increment_skip()
                                    continue
                                
                                # Track result
                                self.result_builder.add_result(result)
                                self.stats.add_model_result(
                                    model=model_name,
                                    category=category,
                                    match=result.match,
                                )
                                
                                if result.match:
                                    self.stats.increment_success()
                                else:
                                    self.stats.increment_failure()
                                
                                category_count += 1
                                
                                # Check limit per category
                                if self.message_limit and category_count >= self.message_limit:
                                    break
                                
                                # Print progress
                                match_sym = "✓" if result.match else "✗"
                                print(f"      ✓ msg {msg_id}: {result.predicted_label} {match_sym}")
                        
                        print(f"      → {category_count} predictions")
                    
                    except Exception as e:
                        print(f"      Error: {e}")
                        traceback.print_exc()
                        continue
        
        # Save and summarize
        output_path = self._finalize()
        return output_path
    
    def _finalize(self) -> str:
        """Save results and print summary"""
        
        # Save
        print(f"\n{'='*70}")
        print(f"  SAVING RESULTS")
        print(f"{'='*70}\n")
        
        output_path = self.result_builder.save()
        print(f"  Saved {len(self.result_builder.results)} results")
        print(f"      {output_path}\n")
        
        # Print stats
        self.stats.print_summary()
        
        # Per-model accuracy
        print(f"{'─'*70}")
        print(f"  PER-MODEL ACCURACY")
        print(f"{'─'*70}\n")
        for model, stats in self.stats.per_model.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {model:15} {accuracy:6.2%} ({stats['correct']}/{stats['total']})")
        
        # Per-category accuracy
        print(f"\n{'─'*70}")
        print(f"  PER-CATEGORY ACCURACY")
        print(f"{'─'*70}\n")
        for category, stats in self.stats.per_category.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {category:20} {accuracy:6.2%} ({stats['correct']}/{stats['total']})")
        
        print(f"\n{'='*70}\n")
        
        return output_path