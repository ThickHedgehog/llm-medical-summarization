#!/bin/bash
# =============================================================================
# Run All Experiments
# =============================================================================
# This script runs all 9 experimental configurations:
# - 3 models (Flan-T5 Small, Base, Large)
# - 3 methods (Full Fine-Tuning, LoRA, Prompt Tuning)
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --debug  # For quick testing
# =============================================================================

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
DEBUG_MODE="${1:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  Medical Summarization Experiments"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Debug mode: ${DEBUG_MODE:-disabled}"
echo ""

# Models to test
MODELS=(
    "google/flan-t5-small"
    "google/flan-t5-base"
    "google/flan-t5-large"
)

# Methods to test
METHODS=(
    "lora"
    "prompt_tuning"
    "full_ft"
)

# Debug flags
DEBUG_FLAGS=""
if [ "$DEBUG_MODE" == "--debug" ]; then
    echo -e "${YELLOW}Running in DEBUG mode with small dataset${NC}"
    DEBUG_FLAGS="--debug --epochs 1"
fi

# Counter for experiments
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#METHODS[@]}))
CURRENT=0
FAILED=0

# Log file
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"
mkdir -p "$OUTPUT_DIR"
echo "Experiment Log - $(date)" > "$LOG_FILE"

# Function to run single experiment
run_experiment() {
    local model=$1
    local method=$2
    
    CURRENT=$((CURRENT + 1))
    local model_short=$(echo "$model" | sed 's/google\///')
    local exp_name="${model_short}_${method}"
    
    echo ""
    echo -e "${YELLOW}=============================================="
    echo "  Experiment $CURRENT/$TOTAL_EXPERIMENTS"
    echo "  Model: $model"
    echo "  Method: $method"
    echo "==============================================${NC}"
    
    # Log start
    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Starting: $exp_name at $(date)" >> "$LOG_FILE"
    
    # Run experiment
    if python scripts/run_experiment.py \
        --model_name "$model" \
        --method "$method" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$exp_name" \
        $DEBUG_FLAGS; then
        
        echo -e "${GREEN}✓ Completed: $exp_name${NC}"
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] Completed: $exp_name at $(date)" >> "$LOG_FILE"
    else
        echo -e "${RED}✗ Failed: $exp_name${NC}"
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] FAILED: $exp_name at $(date)" >> "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
}

# Start time
START_TIME=$(date +%s)

# Run all experiments
echo ""
echo "Starting ${TOTAL_EXPERIMENTS} experiments..."
echo ""

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_experiment "$model" "$method"
    done
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

# Summary
echo ""
echo "=============================================="
echo "  EXPERIMENT SUMMARY"
echo "=============================================="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Successful: $((TOTAL_EXPERIMENTS - FAILED))"
echo "Failed: $FAILED"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

# Generate comparison report
echo ""
echo "Generating comparison report..."
python scripts/generate_report.py --results_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/comparison_report.md"

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Warning: $FAILED experiment(s) failed. Check logs for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All experiments completed successfully!${NC}"
fi
