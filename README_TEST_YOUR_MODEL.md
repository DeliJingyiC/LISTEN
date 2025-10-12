# Test Your Model on LISTEN Benchmark

This guide shows you how to evaluate your own model on the LISTEN benchmark using the simple template script: `scripts/test_your_model.py`

## Available Models

We provide ready-to-run scripts for these models:

| Model | Type | Experiments | Script |
|-------|------|-------------|--------|
| **Your Model** | Template | All (1-4) | `test_your_model.py` |
| Qwen2.5-Omni | Multimodal | All (1-4) | `test_qwen2_5_omni_all.py` |
| Qwen3-instruct | Text-only | 1,2A,3A | `test_qwen3_instruct.py` |
| Qwen3-Omni  | Multimodal | All (1-4) | `test_qwen3_save.py` |
| Baichuan-Omni | Multimodal | All (1-4) | `test_baichuan_omni_all.py` |
| Gemini 2.5 Pro | Multimodal | All (1-4) | `test_gemini2_5_pro.py` |
| Gemini 2.5 Flash | Multimodal | All (1-4) | `test_gemini2_5_flash_all.py` |

**Note**: 
- Experiment 1 variants: `1_text` (text-only), `1_audio` (audio-only), `1_audio_and_text` (multimodal)
- Experiment 2 variants: `2A` (text-only), `2B` (audio-only), `2C` (multimodal)
- Experiment 3 variants: `3A` (text-only), `3B` (audio-only), `3C` (multimodal)
- Experiment 4 variants: `4` (audio-only)

## Quick Start (3 Steps)

### 1. Set Your HuggingFace Token
```python
# Line 27 in scripts/test_your_model.py
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'your_actual_token_here'
```

### 2. Implement Two Functions

Implement these functions (lines 40-92):

```python
def inference_with_audio_clean(audio_array, sr, prompt):
    """Process audio and return answer letter"""
    # Your code here
    # Your messages:
    #messages = [
    #     {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
    #     {
    #         "role": "user", 
    #         "content": [
    #             {"type": "audio", "audio": audio_array},
    #             {"type": "text", "text": prompt},
    #         ]
    #     },
    # ]
    # Apply chat template
    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process multimedia info
    # audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    # print("Finished processing multimedia info")
    
    # Process inputs
    # inputs = processor(
    #     text=text, 
    #     audio=audios, 
    #     images=images, 
    #     videos=videos, 
    #     return_tensors="pt", 
    #     padding=True, 
    #     use_audio_in_video=True
    # )
    inputs = inputs.to(your_model.device)
    print("model.device: ", your_model.device)
    print("Finished processing inputs")
    
    # Generate with your_model parameters
    output = your_model.generate(
       **inputs, 
       use_audio_in_video=True, 
       return_audio=False,
       max_new_tokens=3,
       do_sample=False,  # Deterministic generation
       num_beams=1,      # No beam search
       pad_token_id=processor.tokenizer.eos_token_id,
   )

    print("Finished generating")
    # Decode output
    text_output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text_output

def inference_with_text_clean(text, prompt):
    """Process text and return answer letter"""
    # Your code here
    # Create messages with text input
    # messages = [
    #     {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
    #     {
    #         "role": "user", 
    #         "content": [
    #             {"type": "text", "text": f"Transcription: {text_input}\n\n{prompt}"},
    #         ]
    #     },
    # ]
    
    # Apply chat template
    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    #  Process multimedia info (no audio/video for text-only)
    # audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # Process inputs
    # inputs = processor(
    #     text=text, 
    #     audio=audios, 
    #     images=images, 
    #     videos=videos, 
    #     return_tensors="pt", 
    #     padding=True, 
    #     use_audio_in_video=False
    # )
    inputs = inputs.to(your_model.device)
    
    # Generate with your_model parameters
    output = your_model.generate(
        **inputs, 
        use_audio_in_video=False, 
        return_audio=False,
        max_new_tokens=3,  
    )
    
    # Decode output
    text_output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text_output
```

**That's it!** Just return the letter of the answer (A, B, C, etc.)

### 3. Run

```bash
cd scripts

# Test all experiments
python test_your_model.py --model "YourModel"

# Test specific experiments
python test_your_model.py --model "YourModel" --experiments 2A 3A

# List available experiments
python test_your_model.py --list

# Resume from checkpoint (default) or start fresh
python test_your_model.py --model "YourModel" --no-resume
```


## Running Baseline Models

We provide scripts for all baseline models used in our paper. These are located in `scripts/`:

### Qwen2.5-Omni (All Experiments: 1-4)
```bash
cd scripts
# Run all experiments (1_text, 1_audio, 1_audio_and_text, 2A-C, 3A-C, 4)
sbatch test_qwen2_5_omni-cuda12.8.sh

# Or run specific experiments with Python
python test_qwen2_5_omni_all.py --experiments 1_text 1_audio 2A 3A 4
```

### Qwen3-Omni (Split: Text and Audio)
```bash
cd scripts
# Run all experiments
sbatch test_qwen3_omni-cuda12.8.sh

# Or run separately:
# Text-only experiments (1_text, 2A, 3A)
python test_qwen3_instruct.py --experiments 1_text 2A 3A

# Audio experiments (1_audio, 1_audio_and_text, 2B, 2C, 3B, 3C, 4)
python test_qwen3_save.py --experiments 1_audio 1_audio_and_text 2B 2C 3B 3C 4
```

### Baichuan-Omni (All Experiments: 1-4)
```bash
cd scripts
# Run all experiments (1_text, 1_audio, 1_audio_and_text, 2A-C, 3A-C, 4)
python test_baichuan_omni_all.py --experiments 1_text 1_audio 2A 2B 2C 3A 3B 3C 4
```

### Gemini 2.5 Pro (All Experiments: 1-4)
```bash
cd scripts
# Set your API key first
export GEMINI_API_KEY="your_api_key_here"

# Run all experiments (1_text, 1_audio, 1_audio_and_text, 2A-C, 3A-C, 4)
sbatch test_gemini2_5_pro.sh

# Or run specific experiments with Python
python test_gemini2_5_pro.py --experiments 1_text 2A 3A 4
```

### Gemini 2.5 Flash (All Experiments: 1-4)
```bash
cd scripts
export GEMINI_API_KEY="your_api_key_here"

# Run all experiments (1_text, 1_audio, 1_audio_and_text, 2A-C, 3A-C, 4)
sbatch test_gemini2_5_flash.sh

# Or run specific experiments with Python
python test_gemini2_5_flash_all.py --experiments 1_audio 2B 2C 3B 3C 4
```

**Note**: All baseline scripts include:
- Automatic checkpointing (resume with existing checkpoints)
- Comprehensive metrics (Accuracy, UAR, Macro F1, Micro F1)
- Per-class accuracy breakdown
- Expected accuracy (chance baseline)
- Results saved in `results/` directory

## Experiments

| Experiment | Input | Description |
|------------|-------|-------------|
| 1_text | Text | Neutral-Text Text Only |
| 1_audio | Audio | Neutral-Text Audio Only |
| 1_audio_and_text | Both | Neutral-Text Text + audio |
| 2A | Text | Matched-Emotion Text Only |
| 2B | Audio | Matched-Emotion Audio Only |
| 2C | Both | Matched-Emotion Text + audio |
| 3A | Text | Mismatched-Emotion Text Only |
| 3B | Audio | Mismatched-Emotion Audio Only |
| 3C | Both | Mismatched-Emotion Text + audio |
| 4 | Audio | Paralinguistic Audio Only |

## File Structure

```
LISTEN/
├── scripts/                          # All test scripts
│   ├── test_your_model.py           # Template for your model
│   ├── test_qwen2_5_omni_all.py     # Qwen2.5-Omni 
│   ├── test_qwen3_instruct.py       # Qwen3-instruct
│   ├── test_qwen3_save.py           # Qwen3-Omni 
│   ├── test_baichuan_omni_all.py    # Baichuan-Omni 
│   ├── test_gemini2_5_pro.py        # Gemini 2.5 Pro 
│   ├── test_gemini2_5_flash_all.py  # Gemini 2.5 Flash 
│   ├── evaluation_utils.py          # Shared evaluation code
│   └── *.sh                         # SLURM batch scripts
├── results/                         # Output directory
└── README_TEST_YOUR_MODEL.md        # This file
```

## Results

Results are saved in `results/` folder:
```
results/
├── yourmodel_exp2A_text_20251012_153045.json
├── yourmodel_exp3A_text_20251012_154023.json
└── yourmodel_combined_20251012_160000.json
```

Each file contains:
- Weighted Accuracy, UAR (Unweighted Average Recall)
- Macro F1, Micro F1 scores
- Per-class accuracy breakdown
- Expected accuracy (chance baseline)
- Detailed predictions for each sample

## Tips

1. **Start small**: Test on experiments 2A and 3A first (text-only, smaller)
2. **Use checkpoints**: Script auto-saves every 10 samples, resume if interrupted. If you don't want to resume, use --no-resume
3. **Check output**: Your function must return a single letter: "A", "B", "C", etc.
4. **Handle errors**: Wrap API calls in try-except to handle failures gracefully

## Troubleshooting

**"NotImplementedError"**: You need to implement `inference_with_audio_clean()` and `inference_with_text_clean()` functions

**Empty results**: Make sure your functions return "A", "B", "C", etc. (single letter)

**Low accuracy**: 
- Print the prompt and response to debug
- Check if your model understands the task
- Verify letter extraction is working
- Compare with baseline models in `scripts/`

**HuggingFace token error**: Set your token:
```bash
export HUGGINGFACE_HUB_TOKEN="your_token_here"
# Or set it in the script (line 27)
```

**GPU memory issues**:
- Use smaller batch sizes
- Clear cache between samples: `torch.cuda.empty_cache()`
- Test text-only experiments first (smaller memory footprint)

## Citation

```bibtex
@article{listen2024,
  title={Do Audio LLMs Really LISTEN, or Just Transcribe?
Measuring Lexical vs. Acoustic Emotion Cues Reliance},
  year={2025}
}
```
