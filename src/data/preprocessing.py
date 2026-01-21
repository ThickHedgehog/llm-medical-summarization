"""
Text Preprocessing Utilities
============================

Functions for preprocessing text data for T5-based summarization models.
"""

import logging
from typing import Dict, List, Any, Callable, Optional
import re

logger = logging.getLogger(__name__)


def clean_text(text: str, remove_newlines: bool = True) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_newlines: Whether to replace newlines with spaces
        
    Returns:
        Cleaned text
    """
    # remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # remove newlines if specified
    if remove_newlines:
        text = text.replace('\n', ' ').replace('\r', ' ')
    
    # strip leading/trailing whitespace
    text = text.strip()
    
    return text


def create_preprocessing_function(
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    input_column: str = "article",
    target_column: str = "abstract",
    prefix: str = "summarize: ",
    padding: str = "max_length",
    truncation: bool = True,
) -> Callable:
    """
    Create a preprocessing function for the dataset.
    
    This returns a function that can be used with dataset.map().
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        input_column: Name of the input text column
        target_column: Name of the target text column
        prefix: Prefix to add to inputs (e.g., "summarize: " for T5)
        padding: Padding strategy
        truncation: Whether to truncate sequences
        
    Returns:
        Preprocessing function for dataset.map()
    """
    def preprocess_function(examples: Dict[str, List]) -> Dict[str, Any]:
        """
        Preprocess a batch of examples.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Tokenized examples
        """
        # get inputs and targets
        inputs = examples[input_column]
        targets = examples[target_column]
        
        # clean and add prefix to inputs
        inputs = [prefix + clean_text(inp) for inp in inputs]
        targets = [clean_text(tgt) for tgt in targets]
        
        # tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding=padding,
            truncation=truncation,
            return_tensors=None,
        )
        
        # use text_target parameter for T5 models
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding=padding,
            truncation=truncation,
            return_tensors=None,
        )
        
        # replace padding token id with -100 for loss computation
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    return preprocess_function


def preprocess_function(
    examples: Dict[str, List],
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    prefix: str = "summarize: ",
) -> Dict[str, Any]:
    """
    Standalone preprocessing function (alternative API).
    
    Args:
        examples: Batch of examples
        tokenizer: HuggingFace tokenizer
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        prefix: Task prefix
        
    Returns:
        Tokenized examples
    """
    fn = create_preprocessing_function(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prefix=prefix,
    )
    return fn(examples)


def prepare_dataset_for_training(
    dataset,
    tokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    prefix: str = "summarize: ",
    num_proc: int = 4,
    batched: bool = True,
    remove_columns: Optional[List[str]] = None,
):
    """
    Prepare the full dataset for training.
    
    Args:
        dataset: HuggingFace DatasetDict
        tokenizer: Tokenizer
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        prefix: Task prefix
        num_proc: Number of processes for parallel processing
        batched: Whether to process in batches
        remove_columns: Columns to remove after processing
        
    Returns:
        Tokenized dataset ready for training
    """
    logger.info("Preprocessing dataset...")
    
    # Create preprocessing function
    preprocess_fn = create_preprocessing_function(
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prefix=prefix,
    )
    
    # Determine columns to remove
    if remove_columns is None:
        # Remove all original columns, keep only tokenized outputs
        remove_columns = dataset["train"].column_names
    
    # Apply preprocessing
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=batched,
        num_proc=num_proc,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
    )
    
    logger.info("Dataset preprocessing complete")
    logger.info(f"  Tokenized train samples: {len(tokenized_dataset['train'])}")
    logger.info(f"  Tokenized validation samples: {len(tokenized_dataset['validation'])}")
    logger.info(f"  Tokenized test samples: {len(tokenized_dataset['test'])}")
    
    return tokenized_dataset


def get_sample_for_display(
    dataset,
    tokenizer,
    index: int = 0,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    prefix: str = "summarize: ",
) -> Dict[str, str]:
    """
    Get a single sample with both original and tokenized versions for display.
    
    Args:
        dataset: Original dataset
        tokenizer: Tokenizer
        index: Sample index
        max_input_length: Maximum input length
        max_target_length: Maximum target length
        prefix: Task prefix
        
    Returns:
        Dictionary with original and tokenized text
    """
    sample = dataset[index]
    
    # Original text
    original_input = sample["article"]
    original_target = sample["abstract"]
    
    # Preprocessed input
    preprocessed_input = prefix + clean_text(original_input)
    preprocessed_target = clean_text(original_target)
    
    # Tokenize
    input_tokens = tokenizer(
        preprocessed_input,
        max_length=max_input_length,
        truncation=True,
        return_tensors=None,
    )
    
    target_tokens = tokenizer(
        preprocessed_target,
        max_length=max_target_length,
        truncation=True,
        return_tensors=None,
    )
    
    return {
        "original_input": original_input[:500] + "..." if len(original_input) > 500 else original_input,
        "original_target": original_target,
        "preprocessed_input": preprocessed_input[:500] + "..." if len(preprocessed_input) > 500 else preprocessed_input,
        "preprocessed_target": preprocessed_target,
        "input_token_count": len(input_tokens["input_ids"]),
        "target_token_count": len(target_tokens["input_ids"]),
        "input_truncated": len(input_tokens["input_ids"]) == max_input_length,
        "target_truncated": len(target_tokens["input_ids"]) == max_target_length,
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from dataset_loader import load_pubmed_dataset
    
    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    dataset = load_pubmed_dataset(train_size=100, val_size=20, test_size=20)
    
    # Show sample
    sample_info = get_sample_for_display(dataset["train"], tokenizer, index=0)
    
    print("\n" + "=" * 60)
    print("SAMPLE PREPROCESSING")
    print("=" * 60)
    print(f"\nOriginal input (first 500 chars):\n{sample_info['original_input']}")
    print(f"\nOriginal target:\n{sample_info['original_target']}")
    print(f"\nInput token count: {sample_info['input_token_count']}")
    print(f"Target token count: {sample_info['target_token_count']}")
    print(f"Input truncated: {sample_info['input_truncated']}")
    print(f"Target truncated: {sample_info['target_truncated']}")
    
    tokenized = prepare_dataset_for_training(
        dataset,
        tokenizer,
        max_input_length=512,
        max_target_length=128,
    )
    
    print(f"\nTokenized dataset columns: {tokenized['train'].column_names}")
    print(f"Sample tokenized input shape: {len(tokenized['train'][0]['input_ids'])}")
