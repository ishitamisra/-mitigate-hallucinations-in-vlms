#!/usr/bin/env python3
"""
Script to train/compute steering vectors for LLaVA VCD+ASD model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.llava_vcd_asd import LLaVAVCDASD
from evaluation.hallucination_detector import HallucinationDetector
from training.steering_trainer import SteeringVectorTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train steering vectors for LLaVA VCD+ASD")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="liuhaotian/llava-v1.5-13b",
        help="HuggingFace model name for LLaVA"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ydshieh/coco_dataset_script",
        help="HuggingFace dataset name for COCO"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to use for steering vector computation"
    )
    
    # Steering vector arguments
    parser.add_argument(
        "--truth_threshold",
        type=float,
        default=0.3,
        help="Maximum hallucination score for truth examples"
    )
    parser.add_argument(
        "--halluc_threshold",
        type=float,
        default=0.7,
        help="Minimum hallucination score for hallucination examples"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for saving results"
    )
    parser.add_argument(
        "--steering_vectors_path",
        type=str,
        default=None,
        help="Path to save computed steering vectors"
    )
    parser.add_argument(
        "--steering_data_path",
        type=str,
        default=None,
        help="Path to save steering dataset"
    )
    
    # Training arguments
    parser.add_argument(
        "--optimize_hyperparams",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=20,
        help="Number of hyperparameter optimization trials"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate steering effectiveness"
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=200,
        help="Number of samples for evaluation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default paths if not provided
    if args.steering_vectors_path is None:
        args.steering_vectors_path = os.path.join(args.output_dir, "steering_vectors.pt")
    if args.steering_data_path is None:
        args.steering_data_path = os.path.join(args.output_dir, "steering_data.json")
    
    logger.info("Initializing model and trainer...")
    
    # Initialize model
    model = LLaVAVCDASD(model_name=args.model_name)
    
    # Initialize detector and trainer
    detector = HallucinationDetector()
    trainer = SteeringVectorTrainer(model=model, detector=detector)
    
    logger.info(f"Collecting steering data from {args.num_samples} samples...")
    
    # Collect steering data
    steering_dataset = trainer.collect_steering_data(
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        save_path=args.steering_data_path
    )
    
    logger.info("Computing steering vectors...")
    
    # Compute steering vectors
    positive_vec, negative_vec = trainer.compute_steering_vectors(
        steering_dataset=steering_dataset,
        truth_threshold=args.truth_threshold,
        halluc_threshold=args.halluc_threshold,
        save_path=args.steering_vectors_path
    )
    
    logger.info(f"Steering vectors computed and saved to {args.steering_vectors_path}")
    
    # Hyperparameter optimization
    if args.optimize_hyperparams:
        logger.info("Running hyperparameter optimization...")
        
        optimization_results = trainer.optimize_hyperparameters(
            steering_dataset=steering_dataset,
            num_trials=args.num_trials
        )
        
        # Save optimization results
        import json
        results_path = os.path.join(args.output_dir, "hyperparameter_optimization.json")
        with open(results_path, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_path}")
        logger.info(f"Best parameters: {optimization_results['best_params']}")
        logger.info(f"Best score: {optimization_results['best_score']:.4f}")
    
    # Evaluation
    if args.evaluate:
        logger.info("Evaluating steering effectiveness...")
        
        evaluation_results = trainer.evaluate_steering_effectiveness(
            test_dataset=steering_dataset,
            num_samples=args.eval_samples
        )
        
        # Save evaluation results
        import json
        eval_path = os.path.join(args.output_dir, "steering_evaluation.json")
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_path}")
        logger.info("Evaluation Summary:")
        logger.info(f"  Baseline hallucination rate: {evaluation_results['baseline_hallucination_rate']:.3f}")
        logger.info(f"  Steered hallucination rate: {evaluation_results['steered_hallucination_rate']:.3f}")
        logger.info(f"  Relative improvement: {evaluation_results['relative_improvement']:.1f}%")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()