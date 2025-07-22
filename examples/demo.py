#!/usr/bin/env python3
"""
Demo script showing how to use the LLaVA VCD+ASD model for hallucination mitigation.
"""

import sys
import os
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.llava_vcd_asd import LLaVAVCDASD
from evaluation.hallucination_detector import HallucinationDetector


def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def main():
    print("🚀 LLaVA VCD+ASD Demo")
    print("=" * 50)
    
    # Initialize model
    print("Loading LLaVA model...")
    model = LLaVAVCDASD(
        model_name="liuhaotian/llava-v1.5-13b",
        vcd_alpha=0.1,
        lambda_positive=0.2,
        lambda_negative=0.4
    )
    
    # Initialize hallucination detector
    detector = HallucinationDetector()
    
    # Example image URLs (you can replace these with your own)
    example_images = [
        {
            "url": "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=500&h=400&fit=crop",
            "description": "A cute dog sitting in grass",
            "ground_truth": ["A dog sitting on grass", "A golden retriever in a park"]
        },
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500&h=400&fit=crop",
            "description": "A mountain landscape with lake",
            "ground_truth": ["Mountains reflected in a lake", "A scenic mountain lake view"]
        }
    ]
    
    # Check if we have pre-computed steering vectors
    steering_vectors_path = "outputs/steering_vectors.pt"
    if os.path.exists(steering_vectors_path):
        print(f"Loading pre-computed steering vectors from {steering_vectors_path}")
        model.load_steering_vectors(steering_vectors_path)
        has_steering = True
    else:
        print("⚠️  No pre-computed steering vectors found.")
        print("   Run 'python scripts/train_steering_vectors.py' first for full VCD+ASD functionality.")
        has_steering = False
    
    print("\n" + "=" * 50)
    
    for i, example in enumerate(example_images, 1):
        print(f"\n📸 Example {i}: {example['description']}")
        print("-" * 30)
        
        try:
            # Load image
            print(f"Loading image from: {example['url']}")
            image = load_image_from_url(example['url'])
            
            prompt = "Describe this image in detail."
            
            # Generate with baseline LLaVA
            print("\n🔹 Baseline LLaVA:")
            baseline_text = model.generate_baseline(image, prompt, max_new_tokens=80)
            print(f"Generated: {baseline_text}")
            
            # Compute hallucination score for baseline
            baseline_score = detector.compute_hallucination_score(
                baseline_text, example['ground_truth']
            )
            print(f"Hallucination Score: {baseline_score['hallucination_score']:.3f}")
            print(f"Precision: {baseline_score['precision']:.3f}")
            
            # Generate with VCD only (no steering)
            print("\n🔹 VCD Only:")
            # Temporarily disable steering
            original_pos = model.positive_steering_vector
            original_neg = model.negative_steering_vector
            model.positive_steering_vector = None
            model.negative_steering_vector = None
            
            vcd_text = model.generate_with_vcd_asd(image, prompt, max_new_tokens=80)
            print(f"Generated: {vcd_text}")
            
            vcd_score = detector.compute_hallucination_score(
                vcd_text, example['ground_truth']
            )
            print(f"Hallucination Score: {vcd_score['hallucination_score']:.3f}")
            print(f"Precision: {vcd_score['precision']:.3f}")
            
            # Restore steering vectors
            model.positive_steering_vector = original_pos
            model.negative_steering_vector = original_neg
            
            # Generate with VCD+ASD (if steering vectors available)
            if has_steering:
                print("\n🔹 VCD + ASD:")
                vcd_asd_text = model.generate_with_vcd_asd(image, prompt, max_new_tokens=80)
                print(f"Generated: {vcd_asd_text}")
                
                vcd_asd_score = detector.compute_hallucination_score(
                    vcd_asd_text, example['ground_truth']
                )
                print(f"Hallucination Score: {vcd_asd_score['hallucination_score']:.3f}")
                print(f"Precision: {vcd_asd_score['precision']:.3f}")
                
                # Show improvement
                baseline_halluc = baseline_score['hallucination_score']
                vcd_asd_halluc = vcd_asd_score['hallucination_score']
                if baseline_halluc > 0:
                    improvement = (baseline_halluc - vcd_asd_halluc) / baseline_halluc * 100
                    print(f"📈 Improvement over baseline: {improvement:.1f}%")
            
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")
            continue
    
    print("\n🎉 Demo completed!")
    print("\nTo train steering vectors and get full VCD+ASD functionality:")
    print("   python scripts/train_steering_vectors.py --num_samples 1000 --evaluate")
    print("\nTo run comprehensive evaluation:")
    print("   python scripts/evaluate_model.py --compare_all --num_samples 500")


if __name__ == "__main__":
    main()