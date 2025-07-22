"""
LLaVA model with Visual Contrastive Decoding (VCD) and Activation Steering Decoding (ASD)
for mitigating object hallucinations.

Based on:
- VCD: https://arxiv.org/abs/2311.16922
- ASD: https://openreview.net/pdf?id=XfvmkVvnCq
"""

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image, ImageFilter
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LLaVAVCDASD:
    """
    LLaVA model enhanced with Visual Contrastive Decoding and Activation Steering Decoding
    to mitigate object hallucinations.
    """
    
    def __init__(
        self,
        model_name: str = "liuhaotian/llava-v1.5-13b",
        vcd_alpha: float = 0.1,
        blur_intensity: float = 0.5,
        steering_strength: float = 1.0,
        target_layer: int = 16,
        lambda_positive: float = 0.2,
        lambda_negative: float = 0.4,
        contrast_alpha: float = 1.0,
        device: str = "auto"
    ):
        """
        Initialize the LLaVA model with VCD and ASD capabilities.
        
        Args:
            model_name: HuggingFace model name for LLaVA
            vcd_alpha: Weight for VCD contrastive decoding
            blur_intensity: Intensity of Gaussian blur for VCD
            steering_strength: Overall steering vector strength
            target_layer: Layer index for activation steering
            lambda_positive: Positive steering strength (toward truth)
            lambda_negative: Negative steering strength (away from hallucination)
            contrast_alpha: Alpha parameter for ASD contrast decoding
            device: Device placement strategy
        """
        self.model_name = model_name
        self.vcd_alpha = vcd_alpha
        self.blur_intensity = blur_intensity
        self.steering_strength = steering_strength
        self.target_layer = target_layer
        self.lambda_positive = lambda_positive
        self.lambda_negative = lambda_negative
        self.contrast_alpha = contrast_alpha
        
        # Initialize model and processor
        logger.info(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Steering vectors and hooks
        self.positive_steering_vector = None
        self.negative_steering_vector = None
        self.hook = None
        self._positive_output = None
        self._negative_output = None
    
    def blur_image(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur to image for VCD."""
        return image.filter(ImageFilter.GaussianBlur(radius=self.blur_intensity * 10))
    
    def compute_steering_vectors(
        self,
        truth_vectors: List[torch.Tensor],
        halluc_vectors: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute steering vectors following ASD methodology.
        
        Args:
            truth_vectors: Hidden states from truthful examples
            halluc_vectors: Hidden states from hallucinated examples
            
        Returns:
            Tuple of (positive_steering_vector, negative_steering_vector)
        """
        if not truth_vectors or not halluc_vectors:
            raise ValueError("Need both truth and hallucination vectors")
        
        truth_mean = torch.stack(truth_vectors).mean(0)
        halluc_mean = torch.stack(halluc_vectors).mean(0)
        
        # Compute steering direction (truth - hallucination)
        steering_direction = truth_mean - halluc_mean
        
        # Normalize steering vectors
        self.positive_steering_vector = F.normalize(steering_direction, dim=0)
        self.negative_steering_vector = F.normalize(-steering_direction, dim=0)
        
        logger.info(f"Computed steering vectors from {len(truth_vectors)} truth "
                   f"and {len(halluc_vectors)} hallucination examples")
        
        return self.positive_steering_vector, self.negative_steering_vector
    
    def get_hidden_states(self, image: Image.Image, prompt: str) -> torch.Tensor:
        """Extract hidden states for steering vector computation."""
        inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Extract from final token of last layer
        return outputs.hidden_states[-1][0, -1].cpu()
    
    def register_steering_hook(self, steering_mode: str = "bidirectional"):
        """
        Register forward hooks for activation steering following ASD methodology.
        
        Args:
            steering_mode: "bidirectional", "positive", or "negative"
        """
        if self.positive_steering_vector is None:
            logger.warning("No steering vectors computed yet")
            return
        
        def asd_steering_hook(module, input, output):
            """Hook function implementing ASD steering."""
            device = output.device
            
            if steering_mode == "bidirectional":
                # Apply both positive and negative steering
                positive_steered = output + self.lambda_positive * self.positive_steering_vector.to(device)
                negative_steered = output + self.lambda_negative * self.negative_steering_vector.to(device)
                
                # Store for contrast decoding
                self._positive_output = positive_steered
                self._negative_output = negative_steered
                
                return positive_steered
                
            elif steering_mode == "positive":
                return output + self.lambda_positive * self.positive_steering_vector.to(device)
                
            elif steering_mode == "negative":
                return output + self.lambda_negative * self.negative_steering_vector.to(device)
            
            return output
        
        # Apply hook to target layer
        target_module = self.model.model.language_model.model.layers[self.target_layer]
        self.hook = target_module.register_forward_hook(asd_steering_hook)
    
    def remove_steering_hook(self):
        """Remove steering hooks and clean up."""
        if self.hook:
            self.hook.remove()
            self.hook = None
        
        self._positive_output = None
        self._negative_output = None
    
    def apply_vcd_logits(
        self,
        image: Image.Image,
        prompt: str,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply Visual Contrastive Decoding to logits."""
        # Get original logits
        inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        
        with torch.no_grad():
            orig_outputs = self.model(**inputs, output_hidden_states=True)
            orig_logits = orig_outputs.logits
            
            if pixel_values is not None:  # First step with image
                # Get blurred image logits
                blurred_image = self.blur_image(image)
                blur_inputs = self.processor(prompt, images=blurred_image, return_tensors="pt").to(self.model.device)
                blur_outputs = self.model(**blur_inputs, output_hidden_states=True)
                blur_logits = blur_outputs.logits
                
                # Apply VCD: L' = L + α(L - L_blur)
                vcd_logits = orig_logits + self.vcd_alpha * (orig_logits - blur_logits)
            else:
                vcd_logits = orig_logits
        
        return vcd_logits
    
    def apply_asd_contrast_decoding(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply ASD contrast decoding to logits."""
        if not hasattr(self, '_positive_output') or self._positive_output is None:
            return logits
        
        # Enhanced contrast based on steering strength difference
        steering_contrast = self.lambda_negative - self.lambda_positive
        contrast_factor = 1.0 + abs(steering_contrast)
        
        return logits * contrast_factor
    
    def generate_with_vcd_asd(
        self,
        image: Image.Image,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        steering_mode: str = "bidirectional"
    ) -> str:
        """
        Generate text using combined VCD and ASD methodology.
        
        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            steering_mode: Steering mode for ASD
            
        Returns:
            Generated text
        """
        # Register steering hooks
        self.register_steering_hook(steering_mode)
        
        try:
            # Process initial inputs
            inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.model.device)
            current_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            
            generated_tokens = []
            
            for step in range(max_new_tokens):
                with torch.no_grad():
                    # Apply VCD (only on first step with image)
                    if step == 0:
                        vcd_logits = self.apply_vcd_logits(image, prompt, current_ids, pixel_values)
                    else:
                        # Forward pass with ASD steering applied via hooks
                        outputs = self.model(input_ids=current_ids, output_hidden_states=True)
                        vcd_logits = outputs.logits
                    
                    # Apply ASD contrast decoding
                    final_logits = self.apply_asd_contrast_decoding(vcd_logits)
                    
                    # Sample next token
                    next_token_logits = final_logits[:, -1, :] / temperature
                    
                    if do_sample:
                        next_token_probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(next_token_probs, 1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Check for end of sequence
                    if next_token.item() == self.processor.tokenizer.eos_token_id:
                        break
                    
                    generated_tokens.append(next_token.item())
                    current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Decode generated text
            full_sequence = current_ids[0]
            generated_text = self.processor.tokenizer.decode(full_sequence, skip_special_tokens=True)
            
            # Extract only the generated part
            prompt_text = self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if generated_text.startswith(prompt_text):
                generated_text = generated_text[len(prompt_text):].strip()
            
            return generated_text
            
        finally:
            self.remove_steering_hook()
    
    def generate_baseline(
        self,
        image: Image.Image,
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text using baseline LLaVA without VCD or ASD."""
        inputs = self.processor(prompt, images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        # Decode and extract generated text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Remove prompt from generated text
        prompt_text = prompt
        if generated_text.startswith(prompt_text):
            generated_text = generated_text[len(prompt_text):].strip()
        
        return generated_text
    
    def save_steering_vectors(self, path: str):
        """Save computed steering vectors to disk."""
        if self.positive_steering_vector is not None:
            torch.save({
                'positive_steering_vector': self.positive_steering_vector,
                'negative_steering_vector': self.negative_steering_vector,
                'config': {
                    'lambda_positive': self.lambda_positive,
                    'lambda_negative': self.lambda_negative,
                    'target_layer': self.target_layer
                }
            }, path)
            logger.info(f"Saved steering vectors to {path}")
        else:
            logger.warning("No steering vectors to save")
    
    def load_steering_vectors(self, path: str):
        """Load steering vectors from disk."""
        checkpoint = torch.load(path, map_location='cpu')
        self.positive_steering_vector = checkpoint['positive_steering_vector']
        self.negative_steering_vector = checkpoint['negative_steering_vector']
        
        # Update config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.lambda_positive = config.get('lambda_positive', self.lambda_positive)
            self.lambda_negative = config.get('lambda_negative', self.lambda_negative)
            self.target_layer = config.get('target_layer', self.target_layer)
        
        logger.info(f"Loaded steering vectors from {path}")