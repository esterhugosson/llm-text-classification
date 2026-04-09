# Entry point for the experiment

import sys
import argparse
from typing import Optional, List

from src.experiments.experiment import Experiment
from src.experiments.config import MODELS, CATEGORIES


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Run LLM classification experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models and categories
  python -m src.experiments.runner
  
  # Quick test with just 2 messages per category
  python -m src.experiments.runner --test --limit 2
  
  # Run only GPT-4 on interactional_move
  python -m src.experiments.runner --models gpt4o --categories interactional_move
  
  # Run Claude and LLaMA on multiple categories
  python -m src.experiments.runner --models claude llama3 --categories prompt_type is_followup
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
        default="data/dataset/interactions.json",
        help="Path to interactions file"
    )
    
    parser.add_argument(
        "--ground-truth",
        default="data/process_data/processed_ground_truths.json",
        help="Path to ground truth file"
    )
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        args.models = ["gpt4o"]  # Only GPT-4
        args.limit = args.limit or 2  # Default 2 messages
        print("Running in TEST MODE: GPT-4 only, limited messages\n")
    
    # Create and run experiment
    try:
        experiment = Experiment(
            interactions_path=args.interactions,
            ground_truth_path=args.ground_truth,
            message_limit=args.limit,  # Pass the limit
        )
        
        output_path = experiment.run(
            models=args.models,
            categories=args.categories,
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"   Results: {output_path}")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())