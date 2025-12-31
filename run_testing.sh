#!/bin/bash
# =============================================================================
# MM-UNet FIVES - Test Script
# =============================================================================
# Run testing after training completes
# Usage: chmod +x run_testing.sh && bash run_testing.sh
# =============================================================================

set -e

echo "=============================================="
echo "MM-UNet FIVES - Testing"
echo "=============================================="

# Determine Python command
if command -v python3.10 &> /dev/null; then
    PY="python3.10"
elif command -v python3 &> /dev/null; then
    PY="python3"
else
    PY="python"
fi

# Default checkpoint path
CHECKPOINT=${1:-"./model_store/MM_Net_FIVES/best"}

echo "Using Python: $PY"
echo "Checkpoint: $CHECKPOINT"
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la ./model_store/MM_Net_FIVES/ 2>/dev/null || echo "No checkpoints found"
    exit 1
fi

# Run testing with visualization
echo "Running test with visualizations..."
$PY test_fives.py \
    --checkpoint "$CHECKPOINT" \
    --data_root ./fives_preprocessed \
    --save-vis \
    --batch_size 1

echo ""
echo "=============================================="
echo "Testing Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - Metrics: $CHECKPOINT/test_metrics.json"
echo "  - Visualizations: ./visualization/FIVES/"
echo ""
