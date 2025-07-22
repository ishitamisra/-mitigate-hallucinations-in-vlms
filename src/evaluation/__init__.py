"""
Evaluation utilities for hallucination detection and model performance.
"""

from .hallucination_detector import HallucinationDetector, COCOHallucinationEvaluator

__all__ = ["HallucinationDetector", "COCOHallucinationEvaluator"]