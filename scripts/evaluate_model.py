#!/usr/bin/env python3
"""
Script to evaluate LLaVA VCD+ASD model on COCO dataset for hallucination detection.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.llava_vcd_asd import LLaVAVCDASD
from evaluation.hallucination_detector import HallucinationDetector, COCOHallucinationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA VCD+ASD on COCO dataset")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="liuhaotian/llava-v1.5-13b",
        help="HuggingFace model name for LLaVA"
    )
    parser.add_argument(
        "--steering_vectors_path",
        type=str,
        default=None,
        help="Path to load pre-computed steering vectors"
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
        default=500,
        help="Number of samples to evaluate"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=None,
        help="Custom prompts for evaluation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    # Evaluation modes
    parser.add_argument(
        "--evaluate_baseline",
        action="store_true",
        help="Evaluate baseline LLaVA model"
    )
    parser.add_argument(
        "--evaluate_vcd_only",
        action="store_true",
        help="Evaluate VCD-only model"
    )
    parser.add_argument(
        "--evaluate_vcd_asd",
        action="store_true",
        help="Evaluate VCD+ASD model"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare all methods (baseline, VCD, VCD+ASD)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for saving results"
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Save generated texts to file"
    )
    
    # Model parameters
    parser.add_argument(
        "--vcd_alpha",
        type=float,
        default=0.1,
        help="VCD alpha parameter"
    )
    parser.add_argument(
        "--lambda_positive",
        type=float,
        default=0.2,
        help="ASD positive steering strength"
    )
    parser.add_argument(
        "--lambda_negative",
        type=float,
        default=0.4,
        help="ASD negative steering strength"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default prompts if not provided
    if args.prompts is None:
        args.prompts = [
            "Describe this image in detail.",
            "What objects can you see in this image?",
            "What is happening in this image?"
        ]
    
    # Set evaluation modes
    if args.compare_all:
        args.evaluate_baseline = True
        args.evaluate_vcd_only = True
        args.evaluate_vcd_asd = True
    elif not any([args.evaluate_baseline, args.evaluate_vcd_only, args.evaluate_vcd_asd]):
        # Default to VCD+ASD if no mode specified
        args.evaluate_vcd_asd = True
    
    logger.info("Initializing model and evaluator...")
    
    # Initialize model with custom parameters
    model = LLaVAVCDASD(
        model_name=args.model_name,
        vcd_alpha=args.vcd_alpha,
        lambda_positive=args.lambda_positive,
        lambda_negative=args.lambda_negative
    )
    
    # Load steering vectors if provided
    if args.steering_vectors_path and os.path.exists(args.steering_vectors_path):
        logger.info(f"Loading steering vectors from {args.steering_vectors_path}")
        model.load_steering_vectors(args.steering_vectors_path)
    elif args.evaluate_vcd_asd:
        logger.warning("No steering vectors provided for VCD+ASD evaluation. "
                      "Run train_steering_vectors.py first or provide --steering_vectors_path")
    
    # Initialize detector and evaluator
    detector = HallucinationDetector()
    evaluator = COCOHallucinationEvaluator(detector=detector)
    
    # Load dataset
    logger.info(f"Loading COCO dataset: {args.dataset_name}")
    from datasets import load_dataset
    coco_dataset = load_dataset(args.dataset_name, split=f"{args.split}[:{args.num_samples}]")
    
    evaluation_results = {}
    
    # Evaluate baseline
    if args.evaluate_baseline:
        logger.info("Evaluating baseline LLaVA model...")
        
        class BaselineWrapper:
            def __init__(self, model):
                self.model = model
            
            def generate(self, image, prompt):
                return self.model.generate_baseline(
                    image, prompt, 
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
        
        baseline_model = BaselineWrapper(model)
        baseline_results = evaluator.evaluate_model_on_coco(
            model=baseline_model,
            coco_dataset=coco_dataset,
            num_samples=args.num_samples,
            prompts=args.prompts
        )
        evaluation_results["baseline"] = baseline_results
        
        # Save baseline results
        baseline_path = os.path.join(args.output_dir, "baseline_results.json")
        with open(baseline_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        logger.info(f"Baseline results saved to {baseline_path}")
    
    # Evaluate VCD only
    if args.evaluate_vcd_only:
        logger.info("Evaluating VCD-only model...")
        
        class VCDWrapper:
            def __init__(self, model):
                self.model = model
                # Temporarily disable ASD by clearing steering vectors
                self.original_pos = model.positive_steering_vector
                self.original_neg = model.negative_steering_vector
                model.positive_steering_vector = None
                model.negative_steering_vector = None
            
            def generate(self, image, prompt):
                return self.model.generate_with_vcd_asd(
                    image, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    steering_mode="bidirectional"
                )
            
            def __del__(self):
                # Restore steering vectors
                if hasattr(self, 'original_pos'):
                    self.model.positive_steering_vector = self.original_pos
                if hasattr(self, 'original_neg'):
                    self.model.negative_steering_vector = self.original_neg
        
        vcd_model = VCDWrapper(model)
        vcd_results = evaluator.evaluate_model_on_coco(
            model=vcd_model,
            coco_dataset=coco_dataset,
            num_samples=args.num_samples,
            prompts=args.prompts
        )
        evaluation_results["vcd_only"] = vcd_results
        
        # Save VCD results
        vcd_path = os.path.join(args.output_dir, "vcd_results.json")
        with open(vcd_path, 'w') as f:
            json.dump(vcd_results, f, indent=2)
        logger.info(f"VCD results saved to {vcd_path}")
    
    # Evaluate VCD+ASD
    if args.evaluate_vcd_asd:
        logger.info("Evaluating VCD+ASD model...")
        
        if model.positive_steering_vector is None:
            logger.error("No steering vectors available for VCD+ASD evaluation!")
            return
        
        class VCDASDWrapper:
            def __init__(self, model):
                self.model = model
            
            def generate(self, image, prompt):
                return self.model.generate_with_vcd_asd(
                    image, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    steering_mode="bidirectional"
                )
        
        vcd_asd_model = VCDASDWrapper(model)
        vcd_asd_results = evaluator.evaluate_model_on_coco(
            model=vcd_asd_model,
            coco_dataset=coco_dataset,
            num_samples=args.num_samples,
            prompts=args.prompts
        )
        evaluation_results["vcd_asd"] = vcd_asd_results
        
        # Save VCD+ASD results
        vcd_asd_path = os.path.join(args.output_dir, "vcd_asd_results.json")
        with open(vcd_asd_path, 'w') as f:
            json.dump(vcd_asd_results, f, indent=2)
        logger.info(f"VCD+ASD results saved to {vcd_asd_path}")
    
    # Save all results
    all_results_path = os.path.join(args.output_dir, "all_results.json")
    with open(all_results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    for method_name, method_results in evaluation_results.items():
        logger.info(f"\n{method_name.upper()} Results:")
        
        for prompt, prompt_results in method_results.items():
            logger.info(f"  Prompt: '{prompt}'")
            logger.info(f"    Hallucination Rate: {prompt_results['mean_hallucination_score']:.3f}")
            logger.info(f"    Accuracy: {prompt_results['accuracy']:.3f}")
            logger.info(f"    Precision: {prompt_results['mean_precision']:.3f}")
            logger.info(f"    Recall: {prompt_results['mean_recall']:.3f}")
            logger.info(f"    F1: {prompt_results['mean_f1']:.3f}")
    
    # Compare methods if multiple evaluated
    if len(evaluation_results) > 1:
        logger.info(f"\nCOMPARISON (First prompt: '{args.prompts[0]}'):")
        first_prompt = args.prompts[0]
        
        if "baseline" in evaluation_results and "vcd_asd" in evaluation_results:
            baseline_score = evaluation_results["baseline"][first_prompt]["mean_hallucination_score"]
            vcd_asd_score = evaluation_results["vcd_asd"][first_prompt]["mean_hallucination_score"]
            improvement = (baseline_score - vcd_asd_score) / baseline_score * 100
            
            logger.info(f"  Baseline → VCD+ASD:")
            logger.info(f"    Hallucination: {baseline_score:.3f} → {vcd_asd_score:.3f}")
            logger.info(f"    Improvement: {improvement:.1f}%")
    
    logger.info(f"\nAll results saved to {args.output_dir}/")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()