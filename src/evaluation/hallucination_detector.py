"""
Hallucination detection utilities for evaluating LLaVA model performance.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from typing import List, Set, Dict, Any
import logging
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class HallucinationDetector:
    """
    Detector for object hallucinations in vision-language model outputs.
    """
    
    def __init__(self, coco_objects_path: str = None):
        """
        Initialize the hallucination detector.
        
        Args:
            coco_objects_path: Path to COCO objects vocabulary file
        """
        self.stop_words = set(stopwords.words("english"))
        self.common_objects = self._load_coco_objects(coco_objects_path)
        
    def _load_coco_objects(self, coco_objects_path: str = None) -> Set[str]:
        """Load COCO object vocabulary."""
        if coco_objects_path and os.path.exists(coco_objects_path):
            with open(coco_objects_path, 'r') as f:
                return set(json.load(f))
        
        # Default COCO object categories
        default_objects = {
            # Person and clothing
            "person", "people", "man", "woman", "child", "boy", "girl", "baby",
            "hat", "shirt", "pants", "dress", "shoes", "glasses", "tie", "suit",
            
            # Animals
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "lion", "tiger", "monkey", "rabbit", "pig",
            
            # Vehicles
            "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "ship", "skateboard", "snowboard", "surfboard",
            
            # Outdoor objects
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "tree", "flower", "grass", "mountain", "sky", "cloud", "sun", "moon",
            
            # Sports
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            
            # Kitchen
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake",
            
            # Furniture
            "chair", "couch", "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            
            # Appliances
            "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "scissors", "teddy bear", "hair drier", "toothbrush",
            
            # Common objects
            "bag", "backpack", "umbrella", "handbag", "suitcase", "box",
            "door", "window", "wall", "floor", "ceiling", "light", "lamp",
            "mirror", "picture", "painting", "frame"
        }
        
        logger.info(f"Using default COCO objects vocabulary with {len(default_objects)} objects")
        return default_objects
    
    def extract_objects(self, text: str) -> Set[str]:
        """
        Extract object mentions from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Set of detected objects
        """
        # Preprocess text
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter for objects (not stop words and in object vocabulary)
        objects = set()
        for token in tokens:
            if token not in self.stop_words and token in self.common_objects:
                objects.add(token)
        
        # Also check for multi-word objects
        text_clean = " ".join(tokens)
        for obj in self.common_objects:
            if " " in obj and obj in text_clean:
                objects.add(obj)
        
        return objects
    
    def compute_hallucination_score(
        self,
        generated_text: str,
        ground_truth_captions: List[str]
    ) -> Dict[str, Any]:
        """
        Compute hallucination score for generated text against ground truth.
        
        Args:
            generated_text: Generated caption/description
            ground_truth_captions: List of ground truth captions
            
        Returns:
            Dictionary with hallucination metrics
        """
        # Extract objects from generated text
        gen_objects = self.extract_objects(generated_text)
        
        # Extract objects from all ground truth captions
        gt_objects = set()
        for caption in ground_truth_captions:
            gt_objects |= self.extract_objects(caption)
        
        if not gen_objects:
            return {
                "hallucination_score": 0.0,
                "precision": 1.0,
                "recall": 0.0,
                "f1": 0.0,
                "generated_objects": list(gen_objects),
                "ground_truth_objects": list(gt_objects),
                "hallucinated_objects": [],
                "correct_objects": [],
                "missed_objects": list(gt_objects)
            }
        
        # Compute metrics
        correct_objects = gen_objects & gt_objects
        hallucinated_objects = gen_objects - gt_objects
        missed_objects = gt_objects - gen_objects
        
        # Hallucination score: fraction of generated objects that are hallucinated
        hallucination_score = len(hallucinated_objects) / len(gen_objects)
        
        # Precision: fraction of generated objects that are correct
        precision = len(correct_objects) / len(gen_objects)
        
        # Recall: fraction of ground truth objects that were generated
        recall = len(correct_objects) / len(gt_objects) if gt_objects else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "hallucination_score": hallucination_score,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "generated_objects": list(gen_objects),
            "ground_truth_objects": list(gt_objects),
            "hallucinated_objects": list(hallucinated_objects),
            "correct_objects": list(correct_objects),
            "missed_objects": list(missed_objects)
        }
    
    def evaluate_batch(
        self,
        generated_texts: List[str],
        ground_truth_captions: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of generated texts.
        
        Args:
            generated_texts: List of generated captions
            ground_truth_captions: List of ground truth caption lists
            
        Returns:
            Aggregated evaluation metrics
        """
        if len(generated_texts) != len(ground_truth_captions):
            raise ValueError("Mismatch between generated texts and ground truth")
        
        all_scores = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        detailed_results = []
        
        for gen_text, gt_captions in zip(generated_texts, ground_truth_captions):
            result = self.compute_hallucination_score(gen_text, gt_captions)
            
            all_scores.append(result["hallucination_score"])
            all_precisions.append(result["precision"])
            all_recalls.append(result["recall"])
            all_f1s.append(result["f1"])
            
            detailed_results.append(result)
        
        return {
            "mean_hallucination_score": sum(all_scores) / len(all_scores),
            "mean_precision": sum(all_precisions) / len(all_precisions),
            "mean_recall": sum(all_recalls) / len(all_recalls),
            "mean_f1": sum(all_f1s) / len(all_f1s),
            "accuracy": 1.0 - (sum(all_scores) / len(all_scores)),  # 1 - hallucination_score
            "num_samples": len(generated_texts),
            "detailed_results": detailed_results
        }
    
    def save_coco_objects(self, path: str):
        """Save the COCO objects vocabulary to a file."""
        with open(path, 'w') as f:
            json.dump(list(self.common_objects), f, indent=2)
        logger.info(f"Saved COCO objects vocabulary to {path}")
    
    def load_coco_objects(self, path: str):
        """Load COCO objects vocabulary from a file."""
        with open(path, 'r') as f:
            self.common_objects = set(json.load(f))
        logger.info(f"Loaded COCO objects vocabulary from {path}")


class COCOHallucinationEvaluator:
    """
    Specialized evaluator for COCO dataset hallucination evaluation.
    """
    
    def __init__(self, detector: HallucinationDetector = None):
        """
        Initialize the COCO evaluator.
        
        Args:
            detector: HallucinationDetector instance
        """
        self.detector = detector or HallucinationDetector()
    
    def evaluate_model_on_coco(
        self,
        model,
        coco_dataset,
        num_samples: int = 1000,
        prompts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on COCO dataset for hallucinations.
        
        Args:
            model: Model with generate method
            coco_dataset: COCO dataset
            num_samples: Number of samples to evaluate
            prompts: List of prompts to use
            
        Returns:
            Evaluation results
        """
        if prompts is None:
            prompts = [
                "Describe this image in detail.",
                "What objects can you see in this image?",
                "What is happening in this image?"
            ]
        
        results_by_prompt = {}
        
        for prompt in prompts:
            logger.info(f"Evaluating with prompt: '{prompt}'")
            
            generated_texts = []
            ground_truths = []
            
            for i, example in enumerate(coco_dataset):
                if i >= num_samples:
                    break
                
                # Get image and captions
                image = example["image"]
                captions = example.get("captions", [example.get("caption", "")])
                
                if isinstance(captions, str):
                    captions = [captions]
                
                # Generate text with model
                try:
                    if hasattr(model, 'generate_with_vcd_asd'):
                        generated_text = model.generate_with_vcd_asd(image, prompt)
                    elif hasattr(model, 'generate_baseline'):
                        generated_text = model.generate_baseline(image, prompt)
                    else:
                        # Assume model has a generate method
                        generated_text = model.generate(image, prompt)
                    
                    generated_texts.append(generated_text)
                    ground_truths.append(captions)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate for sample {i}: {e}")
                    continue
            
            # Evaluate batch
            results = self.detector.evaluate_batch(generated_texts, ground_truths)
            results_by_prompt[prompt] = results
            
            logger.info(f"Results for '{prompt}':")
            logger.info(f"  Hallucination Rate: {results['mean_hallucination_score']:.3f}")
            logger.info(f"  Accuracy: {results['accuracy']:.3f}")
            logger.info(f"  Precision: {results['mean_precision']:.3f}")
            logger.info(f"  Recall: {results['mean_recall']:.3f}")
            logger.info(f"  F1: {results['mean_f1']:.3f}")
        
        return results_by_prompt