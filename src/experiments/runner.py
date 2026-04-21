import sys
import argparse

from src.experiments.experiment import Experiment
from src.experiments.config import MODELS, CATEGORIES, validate_config, INTERACTIONS_PATH, GROUND_TRUTH_PATH, STRATEGIES
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
  
  # Run only GPT-4 on interactional_move
  python -m src --models gpt4o --categories interactional_move
  
  # Run Claude on multiple categories
  python -m src --models claude --categories prompt_type is_followup
        """
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
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES.keys()),
        help=f"Strategies to test (default: all). Options: {', '.join(STRATEGIES.keys())}"
    )

    args = parser.parse_args()

    # Validate inputs
    try:
        validate_config()
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1

    # Use all models and categories as default if not specified
    if args.models is None:
        args.models = list(MODELS.keys())
    if args.categories is None:
        args.categories = list(CATEGORIES.keys())
    if args.strategies is None:
        args.strategies = list(STRATEGIES.keys())

    # == Create and run experiment ==
    try:
        logger.info(
            f"Starting experiment with models={args.models}, categories={args.categories}, strategies={args.strategies}")

        experiment = Experiment(
            interactions_path=INTERACTIONS_PATH,
            ground_truth_path=GROUND_TRUTH_PATH,
            message_limit=args.limit,
        )

        # Redo choosen categories to tuple
        category_role_pairs = [
            (cat, role)
            for cat, role in CATEGORIES.items()
              if cat in args.categories 
        ]

        output_path = experiment.run(
            models=args.models,
            categories=category_role_pairs,
            strategies=args.strategies,
        )

        logger.info(f"Experiment completed!")
        logger.info(f"Results saved to: {output_path}")

        return 0

    except DataLoadError as e:
        logger.error(f"Data loading failed: {e}")
        return 1
    except ClassificationError as e:
        logger.error(f"Classification failed: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        log_exception(e, "in experiment execution")
        return 1


if __name__ == "__main__":
    sys.exit(main())
