"""
Data Collator for Seq2Seq Models
================================

Custom data collator for sequence-to-sequence summarization tasks.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForSeq2Seq as HFDataCollatorForSeq2Seq


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator for sequence-to-sequence models.
    
    This wraps the HuggingFace DataCollatorForSeq2Seq with additional
    functionality for our summarization task.
    
    Args:
        tokenizer: The tokenizer used for encoding
        model: The model (optional, used for getting decoder_start_token_id)
        padding: Padding strategy ("longest", "max_length", or True)
        max_length: Maximum length for padding
        pad_to_multiple_of: Pad to a multiple of this value
        label_pad_token_id: Token ID to use for padding labels (-100 by default)
        return_tensors: Type of tensors to return ("pt" for PyTorch)
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    
    def __post_init__(self):
        """Initialize the underlying HuggingFace collator."""
        self.hf_collator = HFDataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            label_pad_token_id=self.label_pad_token_id,
            return_tensors=self.return_tensors,
        )
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched and padded features
        """
        return self.hf_collator(features)


def create_data_collator(
    tokenizer: PreTrainedTokenizerBase,
    model: Optional[Any] = None,
    padding: str = "longest",
    max_length: Optional[int] = None,
) -> DataCollatorForSeq2Seq:
    """
    Factory function to create a data collator.
    
    Args:
        tokenizer: Tokenizer to use
        model: Model (optional)
        padding: Padding strategy
        max_length: Maximum sequence length
        
    Returns:
        Configured DataCollatorForSeq2Seq
    """
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=padding,
        max_length=max_length,
        label_pad_token_id=-100,
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    sample_features = [
        {
            "input_ids": [0, 1, 2, 3, 4],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [0, 1, 2],
        },
        {
            "input_ids": [0, 1, 2],
            "attention_mask": [1, 1, 1],
            "labels": [0, 1, 2, 3, 4, 5],
        },
    ]
    
    collator = create_data_collator(tokenizer)
    batch = collator(sample_features)
    
    print("Collated batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
