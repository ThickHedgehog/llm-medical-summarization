"""
Dataset Loading Utilities
=========================

Functions for loading and preparing the PubMed summarization dataset.
"""

import logging
from typing import Dict, Optional, Tuple, Any

from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

logger = logging.getLogger(__name__)


def load_pubmed_dataset(
    dataset_name: str = "ccdv/pubmed-summarization",
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load the PubMed summarization dataset from HuggingFace.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        train_size: Number of training samples (None for full dataset)
        val_size: Number of validation samples (None for full dataset)
        test_size: Number of test samples (None for full dataset)
        seed: Random seed for reproducible sampling
        cache_dir: Directory to cache the dataset
        
    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Subsample if sizes are specified
    if train_size is not None and train_size < len(dataset["train"]):
        logger.info(f"Subsampling training set to {train_size} samples")
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(train_size))
    
    if val_size is not None and val_size < len(dataset["validation"]):
        logger.info(f"Subsampling validation set to {val_size} samples")
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(val_size))
    
    if test_size is not None and test_size < len(dataset["test"]):
        logger.info(f"Subsampling test set to {test_size} samples")
        dataset["test"] = dataset["test"].shuffle(seed=seed).select(range(test_size))
    
    logger.info(f"Dataset loaded successfully:")
    logger.info(f"  Train: {len(dataset['train'])} samples")
    logger.info(f"  Validation: {len(dataset['validation'])} samples")
    logger.info(f"  Test: {len(dataset['test'])} samples")
    
    return dataset


def get_dataset_statistics(dataset: DatasetDict) -> Dict[str, Any]:
    """
    Compute statistics about the dataset for the report.
    
    Args:
        dataset: The loaded dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "splits": {},
        "input_length": {},
        "target_length": {},
    }
    
    for split_name, split_data in dataset.items():
        # Basic counts
        stats["splits"][split_name] = len(split_data)
        
        # Sample some data for length statistics
        sample_size = min(1000, len(split_data))
        sample = split_data.shuffle(seed=42).select(range(sample_size))
        
        # Input (article) length statistics
        input_lengths = [len(ex["article"].split()) for ex in sample]
        stats["input_length"][split_name] = {
            "mean": sum(input_lengths) / len(input_lengths),
            "min": min(input_lengths),
            "max": max(input_lengths),
            "median": sorted(input_lengths)[len(input_lengths) // 2],
        }
        
        # Target (abstract) length statistics
        target_lengths = [len(ex["abstract"].split()) for ex in sample]
        stats["target_length"][split_name] = {
            "mean": sum(target_lengths) / len(target_lengths),
            "min": min(target_lengths),
            "max": max(target_lengths),
            "median": sorted(target_lengths)[len(target_lengths) // 2],
        }
    
    return stats


def print_dataset_info(dataset: DatasetDict) -> None:
    """Print formatted dataset information."""
    stats = get_dataset_statistics(dataset)
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print("\nSplit Sizes:")
    for split, count in stats["splits"].items():
        print(f"  {split}: {count:,} samples")
    
    print("\nInput (Article) Length (words):")
    for split, lengths in stats["input_length"].items():
        print(f"  {split}: mean={lengths['mean']:.0f}, "
              f"median={lengths['median']:.0f}, "
              f"range=[{lengths['min']}, {lengths['max']}]")
    
    print("\nTarget (Abstract) Length (words):")
    for split, lengths in stats["target_length"].items():
        print(f"  {split}: mean={lengths['mean']:.0f}, "
              f"median={lengths['median']:.0f}, "
              f"range=[{lengths['min']}, {lengths['max']}]")
    
    print("=" * 60 + "\n")


def create_debug_dataset(
    dataset: DatasetDict,
    train_size: int = 100,
    val_size: int = 50,
    test_size: int = 50,
    seed: int = 42,
) -> DatasetDict:
    """
    Create a small debug dataset for quick testing.
    
    Args:
        dataset: Full dataset
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        seed: Random seed
        
    Returns:
        Small DatasetDict for debugging
    """
    return DatasetDict({
        "train": dataset["train"].shuffle(seed=seed).select(range(train_size)),
        "validation": dataset["validation"].shuffle(seed=seed).select(range(val_size)),
        "test": dataset["test"].shuffle(seed=seed).select(range(test_size)),
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Loading PubMed dataset...")
    dataset = load_pubmed_dataset(train_size=1000, val_size=200, test_size=200)
    
    print_dataset_info(dataset)
    
    print("\nSample article (first 500 chars):")
    print(dataset["train"][0]["article"][:500])
    print("\nSample abstract:")
    print(dataset["train"][0]["abstract"])
