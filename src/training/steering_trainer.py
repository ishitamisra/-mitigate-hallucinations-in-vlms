"""
Training utilities for steering vector computation and optimization.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import json
import os
from datasets import load_dataset

from ..models.llava_vcd_asd import LLaVAVCDASD
from ..evaluation.hallucination_detector import HallucinationDetector

logger = logging.getLogger(__name__)


class SteeringDataset(Dataset):
    """Dataset for collecting steering vector training data."""
    
    def __init__(
        self,
        coco_dataset,
        model: LLaVAVCDASD,
        detector: HallucinationDetector,
        num_samples: int = 1000,
        prompts: List[str] = None
    ):
        """
        Initialize steering dataset.
        
        Args:
            coco_dataset: COCO dataset
            model: LLaVA model for generating responses
            detector: Hallucination detector
            num_samples: Number of samples to use
            prompts: List of prompts for generation
        """
        self.coco_dataset = coco_dataset
        self.model = model
        self.detector = detector
        self.num_samples = min(num_samples, len(coco_dataset))
        
        if prompts is None:
            self.prompts = [
                "Describe this image in detail.",
                "What objects can you see in this image?",
                "What is happening in this image?"
            ]
        else:
            self.prompts = prompts
        
        self.data = []
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare training data by generating responses and computing scores."""
        logger.info(f"Preparing steering dataset with {self.num_samples} samples...")
        
        for i in tqdm(range(self.num_samples), desc="Preparing data"):
            example = self.coco_dataset[i]
            image = example["image"]
            
            # Handle different caption formats
            if "captions" in example:
                captions = example["captions"]
            elif "caption" in example:
                captions = [example["caption"]]
            else:
                continue
            
            if isinstance(captions, str):
                captions = [captions]
            
            # Generate responses for each prompt
            for prompt in self.prompts:
                try:
                    # Generate baseline response
                    generated_text = self.model.generate_baseline(image, prompt, max_new_tokens=50)
                    
                    # Compute hallucination score
                    score_result = self.detector.compute_hallucination_score(generated_text, captions)
                    halluc_score = score_result["hallucination_score"]
                    
                    # Get hidden states
                    hidden_states = self.model.get_hidden_states(image, prompt)
                    
                    self.data.append({
                        "image": image,
                        "prompt": prompt,
                        "captions": captions,
                        "generated_text": generated_text,
                        "hallucination_score": halluc_score,
                        "hidden_states": hidden_states,
                        "score_details": score_result
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process sample {i} with prompt '{prompt}': {e}")
                    continue
        
        logger.info(f"Prepared {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_steering_vectors_data(
        self,
        truth_threshold: float = 0.3,
        halluc_threshold: float = 0.7
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract truth and hallucination vectors for steering computation.
        
        Args:
            truth_threshold: Maximum hallucination score for truth examples
            halluc_threshold: Minimum hallucination score for hallucination examples
            
        Returns:
            Tuple of (truth_vectors, hallucination_vectors)
        """
        truth_vectors = []
        halluc_vectors = []
        
        for item in self.data:
            score = item["hallucination_score"]
            hidden_states = item["hidden_states"]
            
            if score <= truth_threshold:
                truth_vectors.append(hidden_states)
            elif score >= halluc_threshold:
                halluc_vectors.append(hidden_states)
        
        logger.info(f"Extracted {len(truth_vectors)} truth vectors and "
                   f"{len(halluc_vectors)} hallucination vectors")
        
        return truth_vectors, halluc_vectors


class SteeringVectorTrainer:
    """Trainer for computing and optimizing steering vectors."""
    
    def __init__(
        self,
        model: LLaVAVCDASD,
        detector: HallucinationDetector = None,
        device: str = None
    ):
        """
        Initialize the steering vector trainer.
        
        Args:
            model: LLaVA model
            detector: Hallucination detector
            device: Device for computations
        """
        self.model = model
        self.detector = detector or HallucinationDetector()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.training_history = []
    
    def collect_steering_data(
        self,
        dataset_name: str = "ydshieh/coco_dataset_script",
        split: str = "validation",
        num_samples: int = 1000,
        prompts: List[str] = None,
        save_path: str = None
    ) -> SteeringDataset:
        """
        Collect data for steering vector computation.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use
            num_samples: Number of samples to collect
            prompts: Prompts for generation
            save_path: Path to save collected data
            
        Returns:
            SteeringDataset with collected data
        """
        logger.info(f"Loading COCO dataset: {dataset_name}")
        coco_dataset = load_dataset(dataset_name, split=f"{split}[:{num_samples}]")
        
        # Create steering dataset
        steering_dataset = SteeringDataset(
            coco_dataset=coco_dataset,
            model=self.model,
            detector=self.detector,
            num_samples=num_samples,
            prompts=prompts
        )
        
        # Save if requested
        if save_path:
            self.save_steering_data(steering_dataset, save_path)
        
        return steering_dataset
    
    def compute_steering_vectors(
        self,
        steering_dataset: SteeringDataset,
        truth_threshold: float = 0.3,
        halluc_threshold: float = 0.7,
        save_path: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute steering vectors from collected data.
        
        Args:
            steering_dataset: Dataset with steering data
            truth_threshold: Threshold for truth examples
            halluc_threshold: Threshold for hallucination examples
            save_path: Path to save computed vectors
            
        Returns:
            Tuple of (positive_steering_vector, negative_steering_vector)
        """
        logger.info("Computing steering vectors...")
        
        # Get truth and hallucination vectors
        truth_vectors, halluc_vectors = steering_dataset.get_steering_vectors_data(
            truth_threshold=truth_threshold,
            halluc_threshold=halluc_threshold
        )
        
        if not truth_vectors or not halluc_vectors:
            raise ValueError("Insufficient data for steering vector computation. "
                           "Try adjusting thresholds or collecting more data.")
        
        # Compute steering vectors using the model
        positive_vec, negative_vec = self.model.compute_steering_vectors(
            truth_vectors, halluc_vectors
        )
        
        # Save if requested
        if save_path:
            self.model.save_steering_vectors(save_path)
        
        return positive_vec, negative_vec
    
    def optimize_hyperparameters(
        self,
        steering_dataset: SteeringDataset,
        validation_dataset: SteeringDataset = None,
        param_ranges: Dict[str, List[float]] = None,
        num_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for VCD and ASD.
        
        Args:
            steering_dataset: Training dataset
            validation_dataset: Validation dataset
            param_ranges: Ranges for hyperparameter search
            num_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters and results
        """
        if param_ranges is None:
            param_ranges = {
                "vcd_alpha": [0.05, 0.1, 0.15, 0.2, 0.25],
                "lambda_positive": [0.1, 0.2, 0.3, 0.4],
                "lambda_negative": [0.2, 0.4, 0.6, 0.8],
                "blur_intensity": [0.3, 0.5, 0.7, 1.0]
            }
        
        if validation_dataset is None:
            validation_dataset = steering_dataset
        
        logger.info(f"Starting hyperparameter optimization with {num_trials} trials...")
        
        best_score = float('inf')
        best_params = {}
        trial_results = []
        
        # Compute steering vectors once
        truth_vectors, halluc_vectors = steering_dataset.get_steering_vectors_data()
        self.model.compute_steering_vectors(truth_vectors, halluc_vectors)
        
        for trial in tqdm(range(num_trials), desc="Hyperparameter optimization"):
            # Sample random hyperparameters
            params = {}
            for param_name, param_range in param_ranges.items():
                params[param_name] = torch.tensor(param_range).uniform_().item() * \
                                   (max(param_range) - min(param_range)) + min(param_range)
            
            # Update model parameters
            original_params = {}
            for param_name, param_value in params.items():
                original_params[param_name] = getattr(self.model, param_name)
                setattr(self.model, param_name, param_value)
            
            # Evaluate on validation set
            try:
                total_score = 0
                num_evaluated = 0
                
                for item in validation_dataset:
                    image = item["image"]
                    prompt = item["prompt"]
                    captions = item["captions"]
                    
                    # Generate with current parameters
                    generated_text = self.model.generate_with_vcd_asd(
                        image, prompt, max_new_tokens=50
                    )
                    
                    # Compute hallucination score
                    result = self.detector.compute_hallucination_score(generated_text, captions)
                    total_score += result["hallucination_score"]
                    num_evaluated += 1
                    
                    # Limit evaluation for speed
                    if num_evaluated >= 100:
                        break
                
                avg_score = total_score / num_evaluated if num_evaluated > 0 else float('inf')
                
                trial_results.append({
                    "trial": trial,
                    "params": params.copy(),
                    "hallucination_score": avg_score
                })
                
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params.copy()
                    logger.info(f"New best score: {best_score:.4f} with params: {best_params}")
                
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                avg_score = float('inf')
            
            # Restore original parameters
            for param_name, param_value in original_params.items():
                setattr(self.model, param_name, param_value)
        
        # Set best parameters
        for param_name, param_value in best_params.items():
            setattr(self.model, param_name, param_value)
        
        logger.info(f"Optimization complete. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "trial_results": trial_results
        }
    
    def evaluate_steering_effectiveness(
        self,
        test_dataset: SteeringDataset,
        num_samples: int = 200
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of computed steering vectors.
        
        Args:
            test_dataset: Test dataset
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results comparing baseline vs. steered generation
        """
        logger.info("Evaluating steering effectiveness...")
        
        baseline_scores = []
        steered_scores = []
        
        num_evaluated = min(num_samples, len(test_dataset))
        
        for i in tqdm(range(num_evaluated), desc="Evaluating"):
            item = test_dataset[i]
            image = item["image"]
            prompt = item["prompt"]
            captions = item["captions"]
            
            try:
                # Baseline generation
                baseline_text = self.model.generate_baseline(image, prompt, max_new_tokens=50)
                baseline_result = self.detector.compute_hallucination_score(baseline_text, captions)
                baseline_scores.append(baseline_result["hallucination_score"])
                
                # Steered generation
                steered_text = self.model.generate_with_vcd_asd(image, prompt, max_new_tokens=50)
                steered_result = self.detector.compute_hallucination_score(steered_text, captions)
                steered_scores.append(steered_result["hallucination_score"])
                
            except Exception as e:
                logger.warning(f"Failed to evaluate sample {i}: {e}")
                continue
        
        # Compute statistics
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        steered_mean = sum(steered_scores) / len(steered_scores)
        
        improvement = (baseline_mean - steered_mean) / baseline_mean * 100
        
        results = {
            "baseline_hallucination_rate": baseline_mean,
            "steered_hallucination_rate": steered_mean,
            "baseline_accuracy": 1.0 - baseline_mean,
            "steered_accuracy": 1.0 - steered_mean,
            "relative_improvement": improvement,
            "absolute_improvement": baseline_mean - steered_mean,
            "num_samples": len(baseline_scores)
        }
        
        logger.info(f"Steering evaluation results:")
        logger.info(f"  Baseline hallucination rate: {baseline_mean:.3f}")
        logger.info(f"  Steered hallucination rate: {steered_mean:.3f}")
        logger.info(f"  Relative improvement: {improvement:.1f}%")
        
        return results
    
    def save_steering_data(self, steering_dataset: SteeringDataset, path: str):
        """Save steering dataset to disk."""
        data_to_save = []
        
        for item in steering_dataset.data:
            # Convert tensors to lists for JSON serialization
            item_copy = item.copy()
            item_copy["hidden_states"] = item["hidden_states"].tolist()
            # Remove PIL image (can't serialize)
            item_copy.pop("image", None)
            data_to_save.append(item_copy)
        
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        logger.info(f"Saved steering data to {path}")
    
    def load_steering_data(self, path: str) -> List[Dict[str, Any]]:
        """Load steering data from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to tensors
        for item in data:
            item["hidden_states"] = torch.tensor(item["hidden_states"])
        
        logger.info(f"Loaded steering data from {path}")
        return data