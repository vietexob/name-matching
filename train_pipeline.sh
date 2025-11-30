#!/usr/bin/env bash

################################################################################
# Name Matching Pipeline Automation Script
#
# This script automates the complete pipeline for training a Name Matching model:
# 1. Generate synthetic names (persons and organizations)
# 2. Create training pairs using Azure OpenAI for alias generation
# 3. Train the LightGBM classifier
#
# Usage: ./train_pipeline.sh [OPTIONS]
# Run ./train_pipeline.sh --help for detailed usage information
################################################################################

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Set PYTHONPATH to project root to ensure imports work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

################################################################################
# Default Configuration
################################################################################

# Step 1: Generate Names
N_PERSONS=700
N_ORGAS=300

# Step 2: Make Dataset
N_NEG=10

# Step 3: Train Model
TEST_SIZE=0.2
THRESH=0.85
TUNE=false
N_TRIALS=25

# General Options
SILENT=false
HUMAN_READABLE=false
DISABLE_TQDM=false
SKIP_GENERATE=false
SKIP_DATASET=false
SKIP_TRAIN=false
VERBOSE=true

################################################################################
# Color Output
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

show_help() {
    cat << EOF
Name Matching Pipeline Automation Script

This script automates the complete training pipeline for Name Matching:
  1. Generate synthetic names (persons and organizations)
  2. Create training pairs using Azure OpenAI for alias generation
  3. Train the LightGBM classifier

USAGE:
    ./train_pipeline.sh [OPTIONS]

OPTIONS:
    Pipeline Control:
    --skip-generate         Skip synthetic name generation step
    --skip-dataset          Skip dataset/alias generation step
    --skip-train            Skip model training step

    Step 1 - Generate Names:
    --n-persons NUM         Number of person names to generate (default: 700)
    --n-orgas NUM           Number of organization names to generate (default: 300)

    Step 2 - Make Dataset:
    --n-neg NUM             Number of negative examples per positive (default: 10)

    Step 3 - Train Model:
    --test-size FLOAT       Test set size as fraction (default: 0.2)
    --thresh FLOAT          Classification threshold for positive class (default: 0.85)
    --tune                  Enable hyperparameter tuning with Optuna
    --n-trials NUM          Number of Optuna trials (default: 25)

    General Options:
    -s, --silent            Enable INFO logging (less verbose)
    -hr, --human-readable   Pretty-print classification report
    -dt, --disable-tqdm     Disable progress bars
    -q, --quiet             Quiet mode (suppress script output, not Python output)
    -h, --help              Show this help message

EXAMPLES:
    # Run full pipeline with defaults:
    ./train_pipeline.sh

    # Generate more names and use higher threshold:
    ./train_pipeline.sh --n-persons 1000 --n-orgas 500 --thresh 0.9

    # Skip name generation and only run dataset + training:
    ./train_pipeline.sh --skip-generate

    # Enable hyperparameter tuning:
    ./train_pipeline.sh --tune --n-trials 50

    # Run in quiet mode:
    ./train_pipeline.sh --quiet

PREREQUISITES:
    - Python environment with required packages installed
    - .env file with Azure OpenAI credentials (for make_dataset step)
    - NLTK stopwords downloaded

ENVIRONMENT VARIABLES (required in .env):
    OPENAI_API_KEY              Legacy OpenAI key
    AZURE_OPENAI_API_VERSION    Azure OpenAI API version
    AZURE_OPENAI_DEPLOYMENT     Azure OpenAI deployment/model name
    AZURE_OPENAI_API_KEY        Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT       Azure OpenAI endpoint URL

OUTPUT:
    - data/raw/mock_sample.csv                          Generated synthetic names
    - data/processed/name_matching_pos_pairs.csv        Positive training pairs
    - data/processed/name_matching_neg_pairs.csv        Negative training pairs
    - models/model_lgb_name_matching.pkl                Trained model
    - models/name_matching_tfidf_ngrams.pkl             TF-IDF vectorizer
    - reports/figures/                                   Performance plots

EOF
}

check_prerequisites() {
    print_step "Checking prerequisites..."

    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    print_info "Python: $(python --version)"

    # Check if .env file exists (needed for make_dataset)
    if [ ! "$SKIP_DATASET" = true ] && [ ! -f ".env" ]; then
        print_error ".env file not found. Required for Azure OpenAI credentials."
        print_info "Create a .env file with the following variables:"
        echo "    OPENAI_API_KEY"
        echo "    AZURE_OPENAI_API_VERSION"
        echo "    AZURE_OPENAI_DEPLOYMENT"
        echo "    AZURE_OPENAI_API_KEY"
        echo "    AZURE_OPENAI_ENDPOINT"
        exit 1
    fi

    # Create necessary directories
    mkdir -p data/raw data/processed models reports/figures
    print_info "Directory structure verified"

    print_success "Prerequisites check passed"
}

################################################################################
# Main Pipeline Functions
################################################################################

run_generate_names() {
    print_header "STEP 1/3: Generating Synthetic Names"

    local cmd="python -m name_matching.data.generate_names"
    cmd="$cmd --n_persons $N_PERSONS"
    cmd="$cmd --n_orgas $N_ORGAS"

    [ "$SILENT" = true ] && cmd="$cmd --silent"
    [ "$HUMAN_READABLE" = true ] && cmd="$cmd --human-readable"
    [ "$DISABLE_TQDM" = true ] && cmd="$cmd --disable-tqdm"

    print_info "Command: $cmd"
    print_info "Generating $N_PERSONS person names and $N_ORGAS organization names..."

    if eval "$cmd"; then
        print_success "Synthetic name generation completed"
        if [ -f "data/raw/mock_sample.csv" ]; then
            local line_count=$(wc -l < "data/raw/mock_sample.csv")
            print_info "Generated file: data/raw/mock_sample.csv ($line_count lines)"
        fi
    else
        print_error "Synthetic name generation failed"
        exit 1
    fi
}

run_make_dataset() {
    print_header "STEP 2/3: Creating Training Pairs (Alias Generation)"

    # Check if synthetic names exist
    if [ ! -f "data/raw/mock_sample.csv" ]; then
        print_error "mock_sample.csv not found. Run generate_names first or use --skip-generate"
        exit 1
    fi

    local cmd="python -m name_matching.data.make_dataset"
    cmd="$cmd --n_neg $N_NEG"

    [ "$SILENT" = true ] && cmd="$cmd --silent"
    [ "$HUMAN_READABLE" = true ] && cmd="$cmd --human-readable"
    [ "$DISABLE_TQDM" = true ] && cmd="$cmd --disable-tqdm"

    print_info "Command: $cmd"
    print_info "Generating positive pairs via Azure OpenAI and $N_NEG negative examples..."
    print_warning "This step may take several minutes depending on the number of names"

    if eval "$cmd"; then
        print_success "Training pair generation completed"
        if [ -f "data/processed/name_matching_pos_pairs.csv" ]; then
            local pos_count=$(wc -l < "data/processed/name_matching_pos_pairs.csv")
            print_info "Positive pairs: data/processed/name_matching_pos_pairs.csv ($pos_count lines)"
        fi
        if [ -f "data/processed/name_matching_neg_pairs.csv" ]; then
            local neg_count=$(wc -l < "data/processed/name_matching_neg_pairs.csv")
            print_info "Negative pairs: data/processed/name_matching_neg_pairs.csv ($neg_count lines)"
        fi
    else
        print_error "Training pair generation failed"
        print_info "Check Azure OpenAI credentials in .env file"
        exit 1
    fi
}

run_train_model() {
    print_header "STEP 3/3: Training Name Matching Model"

    # Check if training pairs exist
    if [ ! -f "data/processed/name_matching_pos_pairs.csv" ] || [ ! -f "data/processed/name_matching_neg_pairs.csv" ]; then
        print_error "Training pairs not found. Run make_dataset first or use --skip-dataset"
        exit 1
    fi

    local cmd="python -m name_matching.models.train_model"
    cmd="$cmd --test-size $TEST_SIZE"
    cmd="$cmd --thresh $THRESH"

    [ "$TUNE" = true ] && cmd="$cmd --tune --n-trials $N_TRIALS"
    [ "$SILENT" = true ] && cmd="$cmd --silent"
    [ "$HUMAN_READABLE" = true ] && cmd="$cmd --human-readable"
    [ "$DISABLE_TQDM" = true ] && cmd="$cmd --disable-tqdm"

    print_info "Command: $cmd"
    print_info "Training LightGBM model with test_size=$TEST_SIZE and threshold=$THRESH..."
    if [ "$TUNE" = true ]; then
        print_info "Hyperparameter tuning enabled with $N_TRIALS trials"
        print_warning "This may take significantly longer"
    fi

    if eval "$cmd"; then
        print_success "Model training completed"
        if [ -f "models/model_lgb_name_matching.pkl" ]; then
            print_info "Model saved: models/model_lgb_name_matching.pkl"
        fi
        if [ -f "models/name_matching_tfidf_ngrams.pkl" ]; then
            print_info "Vectorizer saved: models/name_matching_tfidf_ngrams.pkl"
        fi
        if [ -d "reports/figures" ]; then
            local fig_count=$(find reports/figures -type f -name "*.png" 2>/dev/null | wc -l)
            print_info "Performance plots: reports/figures/ ($fig_count files)"
        fi
    else
        print_error "Model training failed"
        exit 1
    fi
}

################################################################################
# Argument Parsing
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        # Help
        -h|--help)
            show_help
            exit 0
            ;;
        # Pipeline control
        --skip-generate)
            SKIP_GENERATE=true
            shift
            ;;
        --skip-dataset)
            SKIP_DATASET=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        # Step 1 options
        --n-persons)
            N_PERSONS="$2"
            shift 2
            ;;
        --n-orgas)
            N_ORGAS="$2"
            shift 2
            ;;
        # Step 2 options
        --n-neg)
            N_NEG="$2"
            shift 2
            ;;
        # Step 3 options
        --test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        --thresh)
            THRESH="$2"
            shift 2
            ;;
        --tune)
            TUNE=true
            shift
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        # General options
        -s|--silent)
            SILENT=true
            shift
            ;;
        -hr|--human-readable)
            HUMAN_READABLE=true
            shift
            ;;
        -dt|--disable-tqdm)
            DISABLE_TQDM=true
            shift
            ;;
        -q|--quiet)
            VERBOSE=false
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run './train_pipeline.sh --help' for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Main Execution
################################################################################

main() {
    local start_time=$(date +%s)

    if [ "$VERBOSE" = true ]; then
        print_header "Name Matching Training Pipeline"
        echo ""
        echo "Configuration:"
        echo "  Names:      $N_PERSONS persons, $N_ORGAS organizations"
        echo "  Dataset:    $N_NEG negative examples per positive"
        echo "  Training:   test_size=$TEST_SIZE, threshold=$THRESH"
        if [ "$TUNE" = true ]; then
            echo "  Tuning:     Enabled ($N_TRIALS trials)"
        fi
        echo ""
    fi

    # Check prerequisites
    check_prerequisites
    echo ""

    # Step 1: Generate synthetic names
    if [ "$SKIP_GENERATE" = true ]; then
        [ "$VERBOSE" = true ] && print_warning "Skipping synthetic name generation (--skip-generate)"
    else
        run_generate_names
    fi
    echo ""

    # Step 2: Create training pairs
    if [ "$SKIP_DATASET" = true ]; then
        [ "$VERBOSE" = true ] && print_warning "Skipping training pair generation (--skip-dataset)"
    else
        run_make_dataset
    fi
    echo ""

    # Step 3: Train model
    if [ "$SKIP_TRAIN" = true ]; then
        [ "$VERBOSE" = true ] && print_warning "Skipping model training (--skip-train)"
    else
        run_train_model
    fi
    echo ""

    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    if [ "$VERBOSE" = true ]; then
        print_header "Pipeline Completed Successfully"
        print_success "Total runtime: ${minutes}m ${seconds}s"
        echo ""
        echo "Next steps:"
        echo "  • Test the model:     pytest tests/ -v"
        echo "  • Run the API:        python app.py"
        echo "  • Entity resolution:  python entity_resolution.py"
        echo ""
    fi
}

# Run main function
main
