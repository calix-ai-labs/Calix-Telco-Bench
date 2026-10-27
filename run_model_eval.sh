#!/bin/bash
# General 4-GPU parallel model evaluation script
# Supports multiple model configuration files
# Usage: ./run_model_eval.sh <config-file>
# Example: ./run_model_eval.sh config/default.yaml
#
# Note: Ensure .env file exists in the project root with judge credentials:
#   CALIX_TELCO_BENCH_JUDGE_PROVIDER=anthropic
#   CALIX_TELCO_BENCH_JUDGE_MODEL=claude-sonnet-4-5
#   ANTHROPIC_AUTH_TOKEN=<your-token>

set -e  # Exit immediately on error

# Check parameters
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide a configuration file"
    echo ""
    echo "Usage: $0 <config-file>"
    echo ""
    echo "Use a public configuration from config/ or a local configuration from config/local/."
    echo "Example: $0 config/default.yaml"
    echo ""
    exit 1
fi

CONFIG="$1"
VENV_BIN=".venv/bin/calix-telco-bench"

# Check if configuration file exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Configuration file does not exist: $CONFIG"
    exit 1
fi

# Extract model name from config filename (for display)
MODEL_NAME=$(basename "$CONFIG" .yaml)

# Check if .env file exists (required for judging)
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found in project root"
    echo "   The .env file is required for the judging step"
    echo "   Please ensure .env contains:"
    echo "     CALIX_TELCO_BENCH_JUDGE_PROVIDER=anthropic"
    echo "     CALIX_TELCO_BENCH_JUDGE_MODEL=claude-sonnet-4-5"
    echo "     ANTHROPIC_AUTH_TOKEN=<your-token>"
    echo ""
fi

echo "========================================="
echo "Model Evaluation - 4GPU Parallel"
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG"
echo "========================================="
echo ""

# Step 1: Validate dataset
echo "[1/7] Validating dataset integrity..."
$VENV_BIN validate datasets/MANIFEST.json
echo "✅ Dataset validation passed"
echo ""

# Step 2: List route information
echo "[2/7] Checking route distribution..."
$VENV_BIN list-routes datasets/MANIFEST.json
echo ""

# Step 3: Dry-run configuration validation
echo "[3/7] Validating configuration file (dry-run)..."
# Extract all target names from config file
TARGETS=$(grep -A 100 "^targets:" "$CONFIG" | grep "^  [a-z]" | sed 's/:$//' | awk '{print $1}')
TARGET_COUNT=$(echo "$TARGETS" | wc -l)

echo "  Found $TARGET_COUNT evaluation targets"
for target in $TARGETS; do
    echo "  Checking $target..."
    $VENV_BIN run-config --config "$CONFIG" --target "$target" --dry-run > /dev/null
done
echo "✅ Configuration validation passed"
echo ""

# Step 4: Start 4 GPU parallel generation
echo "[4/7] Starting $TARGET_COUNT GPU parallel generation tasks..."
echo "⚠️  This will start $TARGET_COUNT model inference processes in the background"
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Operation cancelled by user"
    exit 1
fi

echo "Starting evaluation..."
$VENV_BIN start-all --config "$CONFIG"
echo "✅ All GPU tasks started"
echo ""

# Step 5: Monitoring command prompt
echo "[5/7] Real-time progress monitoring"
echo "Run the following command in another terminal to monitor progress:"
echo ""
echo "    cd $(pwd)"
echo "    $VENV_BIN tui --config $CONFIG --refresh-seconds 2"
echo ""
echo "Or check status:"
echo "    $VENV_BIN status-all --config $CONFIG"
echo ""

# Step 6: Wait for completion prompt
echo "[6/7] Waiting for generation to complete..."
echo "Generation tasks are running in the background. After completion, run the following commands for judging:"
echo ""
echo "    # Judge all results"
echo "    $VENV_BIN judge-all --config $CONFIG --env-file .env --dry-run  # Validate first"
echo "    $VENV_BIN judge-all-start --config $CONFIG --env-file .env --resume --parallelism 4"
echo ""

# Step 7: Report generation prompt
echo "[7/7] View evaluation results"
echo "After judging is complete, you can view results:"
echo ""
echo "    # View output directory with score summaries"
echo "    ls -la \$(grep 'output_root:' $CONFIG | awk '{print \$2}' | sed 's/\${CALIX_TELCO_BENCH_RELEASE_ROOT}/./g')"
echo ""
echo "    # Each target directory contains:"
echo "    #   - generations.jsonl (model outputs)"
echo "    #   - judgments.jsonl (LLM judge scores)"
echo "    #   - score-summary.json (aggregated scores)"
echo ""

echo "========================================="
echo "✅ Evaluation process started!"
echo "========================================="
echo ""
echo "Quick command reference:"
echo "  • Check status:   $VENV_BIN status-all --config $CONFIG"
echo "  • Real-time monitor:   $VENV_BIN tui --config $CONFIG"
echo "  • Stop all:   $VENV_BIN stop-all --config $CONFIG"
echo ""
