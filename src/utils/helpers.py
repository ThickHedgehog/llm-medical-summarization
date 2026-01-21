"""
Helper Utilities
================

General utility functions for the project.
"""

import logging
import os
import random
from typing import Dict, Any, Optional, Union

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the config
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved config to {output_path}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configurations, with later configs overriding earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    result = {}
    
    for config in configs:
        for key, value in config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    
    return result


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Requested device ("auto", "cuda", "cpu", or specific like "cuda:0")
        
    Returns:
        torch.device object
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    
    return torch.device(device)


def format_number(n: Union[int, float], precision: int = 2) -> str:
    """
    Format a number with appropriate suffix (K, M, B).
    
    Args:
        n: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if n >= 1e9:
        return f"{n / 1e9:.{precision}f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.{precision}f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.{precision}f}K"
    else:
        return f"{n:.{precision}f}"


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_dir(path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_experiment_name(
    model_name: str,
    method: str,
    **kwargs,
) -> str:
    """
    Generate a consistent experiment name.
    
    Args:
        model_name: Name of the model
        method: Adaptation method
        **kwargs: Additional identifiers
        
    Returns:
        Experiment name string
    """
    # Clean model name
    model_short = model_name.split("/")[-1]
    
    parts = [model_short, method]
    
    # Add any additional identifiers
    for key, value in sorted(kwargs.items()):
        if value is not None:
            parts.append(f"{key}={value}")
    
    return "_".join(parts)


def print_experiment_header(
    experiment_name: str,
    model_name: str,
    method: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a formatted experiment header.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Model being used
        method: Adaptation method
        config: Configuration dictionary
    """
    width = 70
    
    print("\n" + "=" * width)
    print(f"{'EXPERIMENT':^{width}}")
    print("=" * width)
    print(f"Name:   {experiment_name}")
    print(f"Model:  {model_name}")
    print(f"Method: {method}")
    
    if config:
        print("-" * width)
        print("Configuration:")
        for key, value in config.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
    
    print("=" * width + "\n")


class DotDict(dict):
    """
    Dictionary that allows attribute-style access.
    """
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{key}'")


def config_to_dotdict(config: Dict[str, Any]) -> DotDict:
    """
    Convert a nested dictionary to DotDict.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DotDict with nested DotDicts
    """
    result = DotDict()
    
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = config_to_dotdict(value)
        else:
            result[key] = value
    
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    set_seed(42)
    print(f"Random number: {random.random()}")
    
    print(f"1234 -> {format_number(1234)}")
    print(f"1234567 -> {format_number(1234567)}")
    print(f"1234567890 -> {format_number(1234567890)}")
    
    print(f"30 seconds -> {format_time(30)}")
    print(f"300 seconds -> {format_time(300)}")
    print(f"7200 seconds -> {format_time(7200)}")
    
    name = get_experiment_name("google/flan-t5-base", "lora", r=16)
    print(f"Experiment name: {name}")
    
    print_experiment_header(
        experiment_name=name,
        model_name="google/flan-t5-base",
        method="lora",
        config={"learning_rate": 3e-4, "epochs": 5},
    )
