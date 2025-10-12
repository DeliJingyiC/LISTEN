

# Load CUDA module (needed for PyTorch)
module load cuda/12.8.1
module load ffmpeg/6.1.1

# Initialize conda properly
echo "Initializing conda..."
source /fs/scratch/PAS1957/delijingyic/miniconda3/bin/activate

# Activate the specific environment
echo "Activating LISTEN-ascend environment..."
conda activate gemini-ascend


# Verify we're using the right Python
echo "Python executable: $(which python)"
echo "Pip executable: $(which pip)"


# Show environment info
echo "Environment info:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "Python path: $PYTHONPATH"
echo "Starting LISTEN experiment run..."


# Parse command line arguments
EXPERIMENTS=""
NO_RESUME=""
FORCE_INPUT_MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --experiments)
            EXPERIMENTS="$2"
            shift 2
            ;;
        --no-resume)
            NO_RESUME="--no-resume"
            shift
            ;;
        --force-input-mode)
            FORCE_INPUT_MODE="--force-input-mode $2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Default experiments if none specified
if [ -z "$EXPERIMENTS" ]; then
    EXPERIMENTS="1_text,1_audio,1_audio_and_text,2A,2B,2C,3A,3B,3C,4"
fi

echo "Running experiments: $EXPERIMENTS"


export GEMINI_API_KEY=" fill in your api key"

python -u /users/PAS2062/delijingyic/project/LISTEN/scripts/test_gemini2_5_flash_all.py \
  --experiments $(echo $EXPERIMENTS | tr ',' ' ') \
  --auto-resume-from-logs \
  --logs-dir /users/PAS2062/delijingyic/project/LISTEN/scripts/output \
  $NO_RESUME \
  $FORCE_INPUT_MODE


echo "Experiment run completed!"