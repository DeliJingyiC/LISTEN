

# Load CUDA module (needed for PyTorch)
module load cuda/12.8.1
module load ffmpeg/6.1.1

# Initialize conda properly
echo "Initializing conda..."
source /fs/scratch/PAS1957/delijingyic/miniconda3/bin/activate

# Activate the specific environment
echo "Activating LISTEN-ascend environment..."
conda activate baichuan_omni

# Verify we're using the right Python
echo "Python executable: $(which python)"
echo "Pip executable: $(which pip)"

# Verify installations
echo "Verifying PyTorch installation..."
python -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test flash-attn
echo "Testing flash-attn..."
python -c "
try:
    import flash_attn
    print(f'flash-attn version: {flash_attn.__version__}')
    print('flash-attn imported successfully')
except Exception as e:
    print(f'flash-attn import failed: {e}')
"

# Show environment info
echo "Environment info:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting LISTEN experiment run..."

# Parse command line arguments
EXPERIMENTS=""
NO_RESUME=""

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
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Default experiments if none specified
if [ -z "$EXPERIMENTS" ]; then
    EXPERIMENTS="1_text,1_audio,1_audio_and_text,2A,2B,2C,3A,3B,3C,4"
    # EXPERIMENTS="4"
fi

echo "Running experiments: $EXPERIMENTS"

python -u /users/PAS2062/delijingyic/project/LISTEN/scripts/test_baichuan_omni_all.py \
  --experiments $(echo $EXPERIMENTS | tr ',' ' ') \
  $NO_RESUME
echo "Experiment run completed!"