# LISTEN: Measuring Lexical vs. Acoustic Emotion Cues Reliance in Audio LLMs

Official repository for the paper: **"Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance"**

📊 **[View Leaderboard](https://delijingyic.github.io/LISTEN-website/)** | 🤗 **[Dataset](https://huggingface.co/datasets/VibeCheck1/LISTEN_full)** | 📄 **[Paper](#citation)**

## Overview

LISTEN is a comprehensive benchmark for evaluating whether audio-capable large language models (LLMs) truly understand acoustic emotion cues or primarily rely on lexical (text-based) information when performing emotion recognition tasks.

### Key Features

- **4 Experiment Types** testing different aspects of audio-text emotion understanding
- **10 Sub-experiments** covering text-only, audio-only, and multimodal inputs
- **Comprehensive Metrics**: Weighted Accuracy, UAR, Macro F1, Micro F1, chance baseline
- **Ready-to-use Scripts** for 6+ state-of-the-art models
- **Easy Integration**: Simple template for testing your own models

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/LISTEN.git
cd LISTEN
```

### 2. Setup Your Model Environment

**Important**: Each model requires its own specific environment setup. Follow the official installation instructions for your model:

- **Qwen2.5-Omni / Qwen3-Omni**: [Qwen2.5-Omni Official Repo](https://github.com/QwenLM/Qwen2.5-Omni), [Qwen3-Omni Official Repo](https://github.com/QwenLM/Qwen3-Omni)
- **Baichuan-Omni**: [Baichuan Official Repo](https://github.com/baichuan-inc/Baichuan-Omni-1.5)
- **Gemini 2.5-Flash**: [Gemini 2.5-Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash)
- **Gemini 2.5-Pro**: [Gemini 2.5-Pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro)
- **Custom Models**: Follow your model's installation guide

```bash
# Example for Qwen models:
pip install transformers accelerate
# ... follow model-specific requirements

# For Gemini models:
pip install google-generativeai
export GEMINI_API_KEY="your_api_key"
```

### 3. Install LISTEN Evaluation Dependencies

```bash
pip install datasets scikit-learn numpy pandas
```

### 4. Set Your HuggingFace Token

```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

### 5. Run Evaluation

```bash
cd scripts
python test_your_model.py --model "YourModel" --experiments 1_text,1_audio,1_audio_and_text,2A,2B,2C,3A,3B,3C,4
```

See [README_TEST_YOUR_MODEL.md](README_TEST_YOUR_MODEL.md) for detailed instructions.

## Experiments

| Experiment | Description | Variants |
|------------|-------------|----------|
| **1** | Neutral-Text | `1_text`, `1_audio`, `1_audio_and_text` |
| **2** | Matched-Emotion | `2A` (text), `2B` (audio), `2C` (both) |
| **3** | Mismatched-Emotion | `3A` (text), `3B` (audio), `3C` (both) |
| **4** | Paralinguistic | `4` (audio-only) |

- **Experiment 1**: Neutral transcriptions - tests emotion recognition on neutral transcription samples
- **Experiment 2**: Matched lexical/acoustic cues - both modalities agree on emotion
- **Experiment 3**: Mismatched lexical/acoustic cues - modalities conflict
- **Experiment 4**: Paralinguistic only - no lexical content (e.g., laughter, sighs)

## Supported Models

We provide ready-to-run evaluation scripts for:

- ✅ Qwen2.5-Omni
- ✅ Qwen3-Instruct
- ✅ Qwen3-Omni 
- ✅ Baichuan-Omni
- ✅ Gemini 2.5 Pro
- ✅ Gemini 2.5 Flash
- ✅ Your Custom Model (template provided)

## Results Format

Each evaluation produces:
```json
{
  "experiment_info": {
    "model_name": "YourModel",
    "experiment_type": "2A",
    "weighted_accuracy": 0.75,
    "uar": 0.72,
    "macro_f1": 0.71,
    "micro_f1": 0.75,
    "expected_accuracy": 0.25
  },
  "per_class_metrics": {...},
  "predictions": [...]
}
```

## Repository Structure

```
LISTEN/
├── README.md                        # This file
├── README_TEST_YOUR_MODEL.md        # Detailed testing guide
├── requirements.txt                 # Python dependencies
└── scripts/
    ├── test_your_model.py          # Template for custom models
    ├── test_qwen2_5_omni_all.py    # Qwen2.5-Omni
    ├── test_qwen3_instruct.py      # Qwen3-Instruct (text)
    ├── test_qwen3_save.py          # Qwen3-Omni (audio)
    ├── test_baichuan_omni_all.py   # Baichuan-Omni
    ├── test_gemini2_5_pro.py       # Gemini 2.5 Pro
    ├── test_gemini2_5_flash_all.py # Gemini 2.5 Flash
    ├── evaluation_utils.py         # Shared evaluation code
    └── *.sh                        # SLURM batch scripts
```

## Documentation

- **[Testing Guide](README_TEST_YOUR_MODEL.md)**: Complete guide for evaluating models
- **Examples**: See baseline model scripts in `scripts/`
- **Metrics**: Comprehensive evaluation including UAR, F1 scores, chance baseline

## Dataset

The LISTEN dataset is hosted on HuggingFace:
- 🤗 [VibeCheck1/LISTEN_full](https://huggingface.co/datasets/VibeCheck1/LISTEN_full)

## Citation

If you use LISTEN in your research, please cite:

```bibtex
@article{listen2025,
  title={Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance},
  year={2025}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact chen.9220@osu.edu.

## Acknowledgments

We thank the developers of the datasets and the models evaluated in this benchmark.

