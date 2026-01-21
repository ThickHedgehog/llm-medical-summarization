# Parameter-Efficient Fine-Tuning for Clinical Text Summarization

A comprehensive comparison of **LoRA**, **Prompt Tuning**, and **Full Fine-Tuning** adaptation methods on the Flan-T5 model family for medical/clinical text summarization.

## Project Overview

This project investigates how different parameter-efficient fine-tuning (PEFT) techniques compare against full fine-tuning for clinical text summarization tasks. We evaluate three model sizes (Flan-T5 Small, Base, Large) across three adaptation strategies, resulting in **9 experimental configurations**.

### Research Questions
1. How do PEFT methods (LoRA, Prompt Tuning) compare to full fine-tuning for clinical summarization?
2. How does model size affect the performance-efficiency trade-off across adaptation methods?
3. What are the practical considerations (memory, training time, parameters) for clinical NLP deployment?

## Project Structure

```
llm-medical-summarization/
├── README.md
├── requirements.txt
├── configs/
│   ├── base_config.yaml          # Base configuration
│   ├── lora_config.yaml          # LoRA-specific settings
│   ├── prompt_tuning_config.yaml # Prompt tuning settings
│   └── full_ft_config.yaml       # Full fine-tuning settings
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py     # Dataset loading utilities
│   │   ├── preprocessing.py      # Text preprocessing
│   │   └── data_collator.py      # Custom data collators
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py       # Model initialization
│   │   ├── lora_wrapper.py       # LoRA configuration
│   │   └── prompt_tuning_wrapper.py  # Prompt tuning setup
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Custom trainer class
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # ROUGE, BERTScore, etc.
│   │   ├── evaluator.py          # Evaluation orchestration
│   │   └── analysis.py           # Results analysis
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py      # Logging configuration
│       ├── memory_tracker.py     # GPU memory tracking
│       └── helpers.py            # General utilities
├── scripts/
│   ├── run_experiment.py         # Main experiment runner
│   ├── run_all_experiments.sh    # Batch experiment script
│   └── generate_report.py        # Generate results report
└── results/                      # Experiment outputs
```

## Experimental Setup

### Models
| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Flan-T5 Small | 80M | `google/flan-t5-small` |
| Flan-T5 Base | 250M | `google/flan-t5-base` |
| Flan-T5 Large | 780M | `google/flan-t5-large` |

### Adaptation Methods
| Method | Trainable Params | Description |
|--------|-----------------|-------------|
| Full Fine-Tuning | 100% | Update all model parameters |
| LoRA | ~0.1-1% | Low-rank adaptation of attention layers |
| Prompt Tuning | ~0.01% | Learnable soft prompts prepended to input |

### Dataset
- **Primary**: [PubMed Summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization)
  - Task: Summarize medical research articles
  - Input: Full article text
  - Output: Abstract/summary
  - Splits: Train (119K), Validation (6.6K), Test (6.7K)

### Evaluation Metrics
- **ROUGE-1, ROUGE-2, ROUGE-L**: N-gram overlap metrics
- **BERTScore**: Semantic similarity using BERT embeddings
- **Training Time**: Wall-clock time per epoch
- **GPU Memory**: Peak memory usage during training
- **Trainable Parameters**: Number of parameters updated

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/eracoding/llm-medical-summarization.git
cd llm-medical-summarization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Single Experiment

```bash
# Full fine-tuning with Flan-T5 Base
python scripts/run_experiment.py \
    --model_name google/flan-t5-base \
    --method full_ft \
    --config configs/full_ft_config.yaml

# LoRA fine-tuning
python scripts/run_experiment.py \
    --model_name google/flan-t5-base \
    --method lora \
    --config configs/lora_config.yaml

# Prompt tuning
python scripts/run_experiment.py \
    --model_name google/flan-t5-base \
    --method prompt_tuning \
    --config configs/prompt_tuning_config.yaml
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

## Expected Results Format

Results are saved to `results/` directory:
```
results/
├── flan-t5-small/
│   ├── full_ft/
│   │   ├── metrics.json
│   │   ├── training_log.csv
│   │   └── predictions.json
│   ├── lora/
│   └── prompt_tuning/
├── flan-t5-base/
└── flan-t5-large/
```

## Key References

1. Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models"
2. Lester et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning"
3. Van Veen et al. (2023). "Clinical Text Summarization: Adapting Large Language Models Can Outperform Human Experts"
4. Suri et al. (2023). "SuryaKiran at MEDIQA-Sum 2023: Leveraging LoRA for Clinical Dialogue Summarization"
5. Tang et al. (2024). "Closing the gap between open source and commercial large language models for medical evidence summarization"

## Report Structure

The final report includes:
1. **Introduction & Dataset**: Task description, dataset statistics
2. **Model & Adaptation Methods**: Technical details of each approach
3. **Training Process**: Hyperparameters, hardware, implementation
4. **Evaluation**: Quantitative metrics and qualitative examples
5. **Comparative Analysis**: Method comparison, efficiency trade-offs

## Hardware Requirements

- **Minimum**: 16GB GPU (for small models with PEFT)
- **Recommended**: 40GB GPU (for all experiments including full FT on large model)
- **Used**: University cluster with 40GB GPU

## License

This project is for educational purposes as part of an LLM course.

## 👤 Author

Ulugbek Shernazarov - Course Project on Large Language Models
