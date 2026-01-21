"""
Memory Tracking Utilities
=========================

Functions for monitoring GPU memory usage.
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import time

import torch

logger = logging.getLogger(__name__)


def get_gpu_memory_info(device_id: int = 0) -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        Dictionary with memory info in GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0,
            "reserved_gb": 0,
            "total_gb": 0,
            "free_gb": 0,
        }
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    free = total - reserved
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": free,
    }


def print_gpu_memory(prefix: str = "", device_id: int = 0) -> None:
    """
    Print current GPU memory usage.
    
    Args:
        prefix: Prefix string for the output
        device_id: CUDA device ID
    """
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available")
        return
    
    info = get_gpu_memory_info(device_id)
    
    print(f"{prefix}GPU Memory: "
          f"Allocated={info['allocated_gb']:.2f}GB, "
          f"Reserved={info['reserved_gb']:.2f}GB, "
          f"Free={info['free_gb']:.2f}GB, "
          f"Total={info['total_gb']:.2f}GB")


@dataclass
class MemorySnapshot:
    """A snapshot of memory state at a point in time."""
    timestamp: float
    label: str
    allocated_gb: float
    reserved_gb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
        }


class MemoryTracker:
    """
    Track GPU memory usage over time.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize the memory tracker.
        
        Args:
            device_id: CUDA device ID to track
        """
        self.device_id = device_id
        self.snapshots: List[MemorySnapshot] = []
        self.start_time = time.time()
    
    def snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            label: Label for this snapshot
            
        Returns:
            MemorySnapshot object
        """
        info = get_gpu_memory_info(self.device_id)
        
        snap = MemorySnapshot(
            timestamp=time.time() - self.start_time,
            label=label,
            allocated_gb=info["allocated_gb"],
            reserved_gb=info["reserved_gb"],
        )
        
        self.snapshots.append(snap)
        return snap
    
    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage from all snapshots.
        
        Returns:
            Dictionary with peak values
        """
        if not self.snapshots:
            return {"peak_allocated_gb": 0, "peak_reserved_gb": 0}
        
        return {
            "peak_allocated_gb": max(s.allocated_gb for s in self.snapshots),
            "peak_reserved_gb": max(s.reserved_gb for s in self.snapshots),
        }
    
    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """
        Get all snapshots as a list of dictionaries.
        
        Returns:
            List of snapshot dictionaries
        """
        return [s.to_dict() for s in self.snapshots]
    
    def print_summary(self) -> None:
        """Print a summary of memory usage."""
        if not self.snapshots:
            print("No memory snapshots recorded.")
            return
        
        peak = self.get_peak_memory()
        
        print("\n" + "=" * 50)
        print("MEMORY USAGE SUMMARY")
        print("=" * 50)
        print(f"Total snapshots: {len(self.snapshots)}")
        print(f"Peak allocated:  {peak['peak_allocated_gb']:.2f} GB")
        print(f"Peak reserved:   {peak['peak_reserved_gb']:.2f} GB")
        
        if len(self.snapshots) > 1:
            print("\nTimeline:")
            for snap in self.snapshots:
                print(f"  [{snap.timestamp:6.1f}s] {snap.label:30s} "
                      f"Alloc={snap.allocated_gb:.2f}GB")
        
        print("=" * 50)
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.snapshots = []
        self.start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)


@contextmanager
def track_memory(label: str = "operation"):
    """
    Context manager to track memory for a block of code.
    
    Args:
        label: Label for this operation
        
    Yields:
        Dictionary that will contain memory stats after the block
    """
    if not torch.cuda.is_available():
        yield {"allocated_before": 0, "allocated_after": 0, "delta": 0}
        return
    
    torch.cuda.synchronize()
    before = get_gpu_memory_info()
    
    stats = {"allocated_before": before["allocated_gb"]}
    
    try:
        yield stats
    finally:
        torch.cuda.synchronize()
        after = get_gpu_memory_info()
        
        stats["allocated_after"] = after["allocated_gb"]
        stats["delta"] = after["allocated_gb"] - before["allocated_gb"]
        
        logger.debug(f"Memory [{label}]: "
                    f"before={before['allocated_gb']:.2f}GB, "
                    f"after={after['allocated_gb']:.2f}GB, "
                    f"delta={stats['delta']:+.2f}GB")


def estimate_model_memory(
    num_parameters: int,
    dtype: str = "float32",
    include_gradients: bool = True,
    include_optimizer: bool = True,
    optimizer_type: str = "adam",
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.
    
    Args:
        num_parameters: Number of model parameters
        dtype: Data type (float32, float16, bfloat16)
        include_gradients: Include gradient storage
        include_optimizer: Include optimizer states
        optimizer_type: Type of optimizer (adam, sgd)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Bytes per parameter for different dtypes
    dtype_bytes = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
    }
    
    bytes_per_param = dtype_bytes.get(dtype, 4)
    model_memory = num_parameters * bytes_per_param
    gradient_memory = model_memory if include_gradients else 0
    
    optimizer_memory = 0
    if include_optimizer:
        if optimizer_type.lower() == "adam":
            optimizer_memory = num_parameters * 4 * 2  # float32
        elif optimizer_type.lower() == "sgd":
            optimizer_memory = num_parameters * 4
    
    total_memory = model_memory + gradient_memory + optimizer_memory
    
    return {
        "model_memory_gb": model_memory / 1024**3,
        "gradient_memory_gb": gradient_memory / 1024**3,
        "optimizer_memory_gb": optimizer_memory / 1024**3,
        "total_memory_gb": total_memory / 1024**3,
        "num_parameters": num_parameters,
        "dtype": dtype,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing memory utilities...\n")
    
    print_gpu_memory("Current: ")
    
    estimates = estimate_model_memory(
        num_parameters=250_000_000,  # flan-t5 base
        dtype="float16",
        include_gradients=True,
        include_optimizer=True,
    )
    
    print("\nMemory estimates for Flan-T5 Base (250M params) with FP16:")
    for key, value in estimates.items():
        if "gb" in key:
            print(f"  {key}: {value:.2f} GB")
        else:
            print(f"  {key}: {value}")
    
    if torch.cuda.is_available():
        tracker = MemoryTracker()
        
        tracker.snapshot("start")
        
        x = torch.randn(1000, 1000, device="cuda")
        tracker.snapshot("after small tensor")
        
        y = torch.randn(10000, 10000, device="cuda")
        tracker.snapshot("after large tensor")
        
        del x, y
        torch.cuda.empty_cache()
        tracker.snapshot("after cleanup")
        
        tracker.print_summary()
