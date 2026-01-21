"""
LLM Medical Summarization Project
=================================

A comprehensive comparison of parameter-efficient fine-tuning methods
for clinical text summarization using the Flan-T5 model family.

Adaptation Methods:
- Full Fine-Tuning
- LoRA (Low-Rank Adaptation)
- Prompt Tuning

Author: Ulugbek Shernazarov
Course: Large Language Models
"""

__version__ = "1.0.0"
__author__ = "Era"

from .data import load_pubmed_dataset, preprocess_function
from .models import load_model_and_tokenizer, setup_lora, setup_prompt_tuning
from .training import run_training
from .evaluation import compute_metrics, evaluate_model

__all__ = [
    "load_pubmed_dataset",
    "preprocess_function",
    "load_model_and_tokenizer",
    "setup_lora",
    "setup_prompt_tuning",
    "run_training",
    "compute_metrics",
    "evaluate_model",
]
