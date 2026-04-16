import sys
import argparse

from src.experiments.experiment import Experiment
from src.experiments.config import MODELS, CATEGORIES, validate_config
from src.utils.logger import setup_logger
from src.utils.error_handler import log_exception, DataLoadError, ClassificationError

logger = setup_logger(__name__)


def main():
    
    parser = argparse.ArgumentParser(
        description="Run LLM classification experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models and categories
  python -m src
  
  # Quick test with just 2 messages per category
  python -m src --test --limit 2
  
  # Run only GPT-4 on interactional_move
  python -m src --models gpt4o --categories interactional_move
  
  # Run Claude on multiple categories
  python -m src --models claude --categories prompt_type is_followup
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (only GPT-4, limited messages)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit messages per category (e.g., --limit 2 for quick test)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help=f"Models to test (default: all). Options: {', '.join(MODELS.keys())}"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORIES.keys()),
        help=f"Categories to test (default: all). Options: {', '.join(CATEGORIES.keys())}"
    )
    
    parser.add_argument(
        "--interactions",
        default="data/process_data/test_interactions.json",
        help="Path to interactions file"
    )
    
    # Define path for ground truth
    parser.add_argument(
        "--ground-truth",
        default="data/process_data/processed_ground_truths.json",
        help="Path to ground truth file"
    )
    
    args = parser.parse_args()
    
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Handle test mode
    if args.test:
        args.models = ["gpt4o"] 
        args.limit = args.limit or 2
        logger.info("Running in TEST MODE: GPT-4 only, limited messages")
    
    # == Create and run experiment ==
    try:
        logger.info(f"Starting experiment with models={args.models}, categories={args.categories}")
        
        experiment = Experiment(
            interactions_path=args.interactions,
            ground_truth_path=args.ground_truth,
            message_limit=args.limit,
        )
        
        output_path = experiment.run(
            models=args.models,
            categories=args.categories,
        )
        
        logger.info(f"Experiment completed!")
        logger.info(f"Results saved to: {output_path}")
        
        return 0
    
    except DataLoadError as e:
        logger.error(f"Data loading failed: {e}")
        print(f"✗ Data loading failed: {e}")
        return 1
    except ClassificationError as e:
        logger.error(f"Classification failed: {e}")
        print(f"✗ Classification failed: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"✗ File not found: {e}")
        return 1
    except Exception as e:
        log_exception(e, "in experiment execution")
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())