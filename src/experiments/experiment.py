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

from src.pipeline.matcher import GroundTruthMatcher
from src.pipeline.classifier_pipeline import ClassificationPipeline
from src.pipeline.result_builder import ResultBuilder

from src.llm.prompt_loader import PromptLoader
from src.views.builder import ExperimentViewBuilder
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        self.view = ExperimentViewBuilder()
        
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
        
        # Print header using view
        self.view.print_experiment_header(models, categories, datetime.now())
        
        # Loop: model → category → strategy → messages
        for model_idx, model_name in enumerate(models, 1):
            self.view.print_model_header(model_name, model_idx, len(models))
            
            try:
                classifier = self._get_classifier(model_name)
            except ValueError as e:
                logger.error(f"Failed to get classifier for {model_name}: {e}")
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
                self.view.print_category_header(category, cat_idx, len(CATEGORIES), role_label)
                
                for strat_idx, strategy in enumerate(STRATEGIES, 1):
                    self.view.print_strategy_header(strategy, strat_idx, len(STRATEGIES))
                    
                    try:
                        self._classify_category(model_name, category, role_filter, strategy, pipeline)
                    except Exception as e:
                        logger.error(f"Error in classification: {e}", exc_info=True)
                        continue
        
        # Save and summarize
        output_path = self._finalize()
        return output_path
    
    def _classify_category(self, model_name, category, role_filter, strategy, pipeline):
        """Classify all messages for a category+strategy using pipeline"""
        results, category_count = pipeline.classify_category(
            interactions=self.interactions,
            category=category,
            strategy=strategy,
            model_name=model_name,
            role_filter=role_filter,
            message_limit=self.message_limit
        )
        
        # Track results
        for result in results:
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
            
            # Display progress using view
            self.view.print_classification_progress(result.message_id, result.predicted_label, result.match)
        
        self.view.print_category_summary(category_count)
    
    def _finalize(self) -> str:
        """Save results and print summary"""
        
        # Save results
        self.view.print_saving_header()
        output_path = self.result_builder.save()
        self.view.print_save_complete(len(self.result_builder.results), output_path)
        
        # Print statistics
        self.view.print_statistics(self.stats)
        self.view.print_per_model_accuracy(self.stats)
        self.view.print_per_category_accuracy(self.stats)
        
        logger.info(f"Experiment finalized with {len(self.result_builder.results)} results")
        logger.info(f"Results saved to: {output_path}")
        
        return output_path