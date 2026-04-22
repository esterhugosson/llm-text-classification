from datetime import datetime
from typing import Optional, List

from src.llm.gpt_4o import LLMClassifierGpt4o
from src.llm.claude_sonnet import LLMClassifierClaudeSonnet
from src.llm.llama_3 import LLMClassifierLlama3

from src.experiments.stats import ExperimentStats

from src.data.loaders.ground_truth_loader import load_ground_truths
from src.data.loaders.interaction_loader import load_interactions

from src.pipeline.matcher import GroundTruthMatcher
from src.pipeline.classifier_pipeline import ClassificationPipeline
from src.pipeline.result_builder import ResultBuilder

from src.llm.prompt_loader import PromptLoader
from src.views.builder import ExperimentViewBuilder
from src.utils.logger import get_logger
from src.utils.error_handler import DataLoadError, ClassificationError, log_exception

logger = get_logger(__name__)

# Main experiment orchestration

class Experiment:
    def __init__(
        self,
        interactions_path,
        ground_truth_path,
        message_limit: Optional[int] = None,
    ):
        # Data paths and limits
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
        logger.info("Starting data loading...")
        
        try:
            # Load interactions and ground truths
            self.interactions = load_interactions(self.interactions_path)
            self.ground_truths = load_ground_truths(self.ground_truth_path)
            
            # Run matcher which will match messages in interactions to existed classified ground truths.
            try:
                self.matcher = GroundTruthMatcher(self.ground_truths)
                logger.debug("Ground truth matcher initialized")
            except Exception as e:
                msg = f"Failed to initialize ground truth matcher: {str(e)}"
                logger.error(msg)
                raise DataLoadError(msg) from e
            
            # Count and validate messages and ground truths
            total_messages = sum(len(msgs) for msgs in self.interactions.values())
            
            if total_messages == 0:
                msg = "No messages found in interactions data"
                logger.warning(msg)
            
            if len(self.ground_truths) == 0:
                msg = "No ground truth labels found"
                logger.warning(msg)
            
            logger.info(f"Data loaded: {len(self.interactions)} threads, {total_messages} messages")
            logger.info(f"Ground truth: {len(self.ground_truths)} threads")
            
        except DataLoadError:
            # Already logged by loader
            raise
        except Exception as e:
            log_exception(e, "during data loading")
            raise DataLoadError(f"Unexpected error during data loading: {str(e)}") from e
    
    def _get_classifier(self, model_name: str):
        """Get LLM classifier instance with error handling"""
        try:
            if model_name == "gpt4o":
                return LLMClassifierGpt4o()
            elif model_name == "claude":
                return LLMClassifierClaudeSonnet()
            elif model_name == "llama3":
                return LLMClassifierLlama3()
            else:
                msg = f"Unknown model: {model_name}"
                logger.error(msg)
                raise ClassificationError(msg)
        except ClassificationError:
            raise
        except Exception as e:
            msg = f"Failed to initialize classifier for {model_name}: {str(e)}"
            logger.error(msg)
            raise ClassificationError(msg) from e
    
    def run(
        self,
        models: Optional[List[str]] = None,
        categories: Optional[List[tuple]] = None,
        strategies: Optional[List[str]] = None
    ) -> str:
        
        # Load data
        try:
            self._load_data()
        except DataLoadError as e:
            logger.error(f"Cannot proceed: {str(e)}")
            raise

        print(models, categories, strategies)

        
        # Print header using view
        category_names = [cat for cat, role in categories]
        self.view.print_experiment_header(models, category_names, datetime.now())
        
        # Loop: model → category → strategy → messages
        # 1: MODEL
        for model_idx, model_name in enumerate(models, 1):
            # Print header
            self.view.print_model_header(model_name, model_idx, len(models))
            
            try:
                # Get regarding classifier
                classifier = self._get_classifier(model_name)
            except ClassificationError as e:
                logger.error(f"Skipping model {model_name}: {str(e)}")
                continue
            
            # Create pipeline for this model
            try:
                pipeline = ClassificationPipeline(
                    classifier=classifier,
                    matcher=self.matcher,
                    prompt_loader=self.prompt_loader,
                )
                logger.debug(f"Pipeline created for model {model_name}")
            except Exception as e:
                msg = f"Failed to create pipeline for {model_name}: {str(e)}"
                logger.error(msg)
                log_exception(e, f"in pipeline creation for {model_name}")
                continue
            
            # 2: CATEGORY
            for cat_idx, (category, role_filter) in enumerate(categories, 1):

                # For the the role filter, give them a name
                role_label = {0: "teacher", 1: "chatbot", None: "all"}.get(
                    role_filter, role_filter 
                )
                # Print header
                self.view.print_category_header(category, cat_idx, len(categories), role_label)
                
                # 3: STRATEGY
                for strat_idx, strategy in enumerate(strategies, 1):
                    # Print header
                    self.view.print_strategy_header(strategy, strat_idx, len(strategies))
                    
                    try:
                        # 4: MESSAGES
                        self._classify_category(model_name, category, role_filter, strategy, pipeline)
                    except ClassificationError as e:
                        logger.error(f"Error classifying {category} with {strategy}: {str(e)}")
                        continue
                    except Exception as e:
                        log_exception(e, f"during {model_name}/{category}/{strategy}")
                        continue
        
        # Save and summarize
        output_path = self._finalize()
        return output_path
    
    # Helper method for classiffication of specific messages 
    def _classify_category(self, model_name, category, role_filter, strategy, pipeline):
        try:
            # Using the pip
            results, category_count = pipeline.classify_category(
                interactions=self.interactions,
                category=category,
                strategy=strategy,
                model_name=model_name,
                role_filter=role_filter,
                message_limit=self.message_limit
            )
            
        except Exception as e:
            msg = f"Pipeline failed for {model_name}/{category}/{strategy}: {str(e)}"
            logger.error(msg)
            log_exception(e, f"in pipeline.classify_category for {category}")
            raise ClassificationError(msg) from e
        
        # Track results with error handling per result
        for result in results:
            try:
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
                
            except Exception as e:
                logger.warning(f"Failed to record result for message {result.message_id}: {str(e)}")
                # Continue processing other results
                continue
        
        self.view.print_category_summary(category_count)
    
    def _finalize(self) -> str:
        """Save results and print summary with error handling"""
        try:
            # Save results
            self.view.print_saving_header()
            output_path = self.result_builder.save()
            
            if not output_path:
                raise Exception("Result builder returned empty path")
            
            logger.info(f"Results saved successfully to {output_path}")
            self.view.print_save_complete(len(self.result_builder.results), output_path)
        except Exception as e:
            msg = f"Failed to save results: {str(e)}"
            logger.error(msg)
            log_exception(e, "in result saving")
            raise
        
        try:
            # Print statistics
            self.view.print_statistics(self.stats)
            self.view.print_per_model_accuracy(self.stats)
            self.view.print_per_category_accuracy(self.stats)
            logger.info(f"Experiment finalized with {len(self.result_builder.results)} results")
        except Exception as e:
            logger.error(f"Failed to generate statistics: {str(e)}")
            log_exception(e, "in statistics generation")
        
        return output_path