#!/usr/bin/env python3
"""
Baichuan-Omni test for VibeCheck experiments
Adapted from the Qwen2.5-Omni test script with Baichuan-Omni specific implementations
"""

# ==============================
# 1. Standard libraries
# ==============================
import os, sys, json, re, glob, shutil, random, hashlib, argparse, importlib.util
from datetime import datetime
from evaluation_utils import calculate_comprehensive_metrics, print_comprehensive_results

# ==============================
# 2. Third-party libs
# ==============================
import numpy as np
import pandas as pd
import torch
import torchaudio
import ujson
import gc

# Hugging Face BEFORE Baichuan imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ==============================
# 3. Dataset preview
# ==============================
# print("🔍 Checking dataset availability before Baichuan imports...")
# try:
#     _preview = load_dataset("VibeCheck1/VibeCheckAudio_final")
#     print(" Dataset loaded:", _preview)
# except Exception as e:
#     print(" Could not pre-load dataset:", e)

# ==============================
# 4. Baichuan-specific imports (aliased to avoid shadowing HF)
# ==============================
BAICHUAN_BASE = "/users/PAS2062/delijingyic/project/VibeCheckBench/Baichuan-Omni-1.5"

# generation.py
spec = importlib.util.spec_from_file_location(
    "baichuan_generation", os.path.join(BAICHUAN_BASE, "web_demo", "generation.py")
)
baichuan_generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baichuan_generation)
decode_wave_vocoder = baichuan_generation.decode_wave_vocoder
GenerationAudioTokens = baichuan_generation.GenerationAudioTokens


# Add hifigan to Python path before importing cosy24k_vocoder
hifigan_path = os.path.join(BAICHUAN_BASE, "third_party", "cosy24k_vocoder")
if hifigan_path not in sys.path:
    sys.path.insert(0, hifigan_path)

# cosy24k_vocoder.py
spec = importlib.util.spec_from_file_location(
    "baichuan_vocoder", os.path.join(BAICHUAN_BASE, "third_party", "cosy24k_vocoder", "cosy24k_vocoder.py")
)
baichuan_vocoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baichuan_vocoder)
Cosy24kVocoder = baichuan_vocoder.Cosy24kVocoder

# constants.py
spec = importlib.util.spec_from_file_location(
    "baichuan_constants", os.path.join(BAICHUAN_BASE, "web_demo", "constants.py")
)
baichuan_constants = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baichuan_constants)
role_prefix = baichuan_constants.role_prefix
COSY_VOCODER = baichuan_constants.COSY_VOCODER

# ==============================
# 5. HuggingFace auth token
# ==============================

# Set HuggingFace token for dataset access
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'fill in your token'

def save_checkpoint(checkpoint_file, results, sample_idx, experiment_info):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'experiment_info': experiment_info,
        'results': results,
        'last_processed_idx': sample_idx,
        'timestamp': datetime.now().isoformat(),
        'total_processed': len(results)
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    
    # Save to temporary file first, then rename (atomic operation)
    temp_file = checkpoint_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, default=str)
    
    # Atomic rename
    shutil.move(temp_file, checkpoint_file)
    print(f"Checkpoint saved: {len(results)} samples processed")

def load_checkpoint(checkpoint_file):
    """Load checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"Found checkpoint: {checkpoint_data['total_processed']} samples already processed")
            print(f"Last checkpoint: {checkpoint_data['timestamp']}")
            
            return (
                checkpoint_data.get('results', []),
                checkpoint_data.get('last_processed_idx', -1),
                checkpoint_data.get('experiment_info', {})
            )
        except Exception as e:
            print(f"Warning: Error loading checkpoint: {e}")
            print("Starting fresh...")
            return [], -1, {}
    
    return [], -1, {}

def get_checkpoint_filename(experiment_type, input_mode, model_prefix="baichuan_omni"):
    """Generate consistent checkpoint filename"""
    return f"results/{model_prefix}_experiment{experiment_type}_{input_mode}_checkpoint_3allemo.json"

def get_corrected_features():
    """Get corrected features schema to fix 'List' -> 'Sequence' issue"""
    from datasets import Features, Value, Sequence, Audio
    
    return Features({
        'id': Value('string'),
        'question': Value('string'),
        'answer': Value('string'),
        'choices': Sequence(Value('string')),  # Fix: was 'List' in original
        'conversation_format': Value('string'),
        'acted_or_spontaneous': Value('string'),
        'sarcasm': Value('bool'),
        'age': Value('string'),
        'gender': Value('string'),
        'transcription': Value('string'),
        'experiment_type': Value('string'),
        'dataset_source': Value('string'),
        'implicit_emotion': Value('string'),
        'explicit_emotion': Value('string'),
        'audio': Audio(sampling_rate=16000)
    })

def load_dataset_with_schema_fix(dataset_name):
    """Load dataset with schema fix for 'List' -> 'Sequence' issue"""
    try:
        # Load with custom features to fix the 'List' -> 'Sequence' issue
        features = get_corrected_features()
        ds = load_dataset(dataset_name, features=features)
        print(f"Loaded dataset: {dataset_name} (with schema fix)")
        return ds
    except Exception as e:
        print(f" Could not load {dataset_name} with schema fix: {e}")
        try:
            # Fallback: try loading without features and handle the error
            print("Trying fallback loading method...")
            ds = load_dataset(dataset_name)
            print(f"Loaded dataset: {dataset_name} (fallback)")
            return ds
        except Exception as e2:
            print(f" Could not load {dataset_name}: {e2}")
            raise e2

def anonymize_sample_id(sample_id):
    """Create an anonymous ID that doesn't leak emotion information"""
    # Use hash to create anonymous but consistent ID
    hash_obj = hashlib.md5(sample_id.encode())
    return f"SAMPLE_{hash_obj.hexdigest()[:8]}"

def clean_sample_data(sample):
    cleaned_sample = sample.copy()
    
    # Replace the ID with anonymous version
    anonymized_id = anonymize_sample_id(sample['id'])
    
    # Keep only essential fields for the task
    essential_fields = ['question', 'choices', 'audio', 'answer', 'transcription']
    
    # Create clean sample with only essential fields plus anonymized ID
    clean_sample = {}
    for key in essential_fields:
        clean_sample[key] = cleaned_sample[key]
    
    # Add the anonymized ID
    clean_sample['id'] = anonymized_id
    
    # Fill other fields with None to maintain structure if needed
    for key in cleaned_sample.keys():
        if key not in clean_sample:
            clean_sample[key] = None
    
    return clean_sample

def randomize_choices(sample, input_mode="audio", original_sample=None, experiment_type=None):
    """Randomize the order of choices to prevent position bias"""
    choices = sample['choices'].copy()
    
    # Determine correct answer based on input mode and experiment type
    if input_mode == "text" and experiment_type in ["2A", "3A"]:
        correct_answer = original_sample.get('explicit_emotion', sample['answer'])
    else:
        correct_answer = sample['answer']
    
    # Skip samples with empty answers
    if not correct_answer or correct_answer.strip() == '':
        print(f"Warning: Empty answer found, skipping randomization for this sample")
        return sample, None
    
    # Only add additional emotions for experiments 3A, 3B, 3C
    if experiment_type in ["3A", "3B", "3C"]:
        additional_emotions = [
            "neutral",
            "sadness", 
            "surprise",
            "happiness",
            "fear",
            "anger",
            "excitement",
            "frustration"
        ]
        
        # Combine original choices with additional emotions and remove duplicates
        all_choices = list(set(choices + additional_emotions))
    else:
        # For other experiments, use original choices only
        all_choices = choices
    
    # Find original position in the choices
    try:
        original_position = all_choices.index(correct_answer)
    except ValueError:
        print(f"Error: Answer '{correct_answer}' not found in choices {all_choices}")
        return sample, None
    
    # Create randomized mapping
    indices = list(range(len(all_choices)))
    random.shuffle(indices)
    
    # Create new choices in randomized order
    randomized_choices = [all_choices[i] for i in indices]
    
    # Find new position of correct answer
    new_position = indices.index(original_position)
    new_letter = chr(65 + new_position)
    
    # Create modified sample
    randomized_sample = sample.copy()
    randomized_sample['choices'] = randomized_choices
    
    return randomized_sample, new_letter

def init_baichuan_model():
    """Initialize Baichuan-Omni model and vocoder with memory optimizations"""
    print("Loading Baichuan-Omni model...")
    
    # Clear memory before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Load vocoder
    vocoder = Cosy24kVocoder.from_pretrained(os.path.join(COSY_VOCODER, "hift.pt"))
    vocoder = vocoder.cuda()
    MODEL_PATH = "/fs/scratch/PAS1957/delijingyic/VibeCheck/Baichuan-Omni-1d5"

    # Load model and tokenizer with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.training = False
    model.bind_processor(tokenizer, training=False, relative_path="/")
    
    # Get special tokens
    audio_start_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
    audio_end_token = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)
    
    print("Baichuan-Omni model loaded successfully")
    return model, tokenizer, vocoder, audio_start_token, audio_end_token

def prepare_audio_for_baichuan(audio_array, sample_rate=16000):
    """Prepare audio data for Baichuan-Omni model"""
    try:
        # Convert to tensor if numpy array
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float()
        else:
            audio_tensor = audio_array.float()
        
        # Ensure correct shape (channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Use Baichuan's expected sample rate (24kHz)
        target_sampling_rate = 24000
        if sample_rate != target_sampling_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sampling_rate)
        
        # Save to temporary file for Baichuan processing
        temp_dir = "/tmp/baichuan_audio"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_audio_{random.randint(10000, 99999)}.wav")
        
        torchaudio.save(temp_path, audio_tensor, target_sampling_rate)
        
        return temp_path
    except Exception as e:
        print(f"Error preparing audio for Baichuan: {e}")
        return None

def inference_with_audio_clean(model, tokenizer, audio_array, sr, prompt, sys_prompt="You are a helpful AI assistant."):
    try:
        # Save numpy array to wav file for Baichuan (it needs file paths, not arrays)
        audio_path = prepare_audio_for_baichuan(audio_array, sr)
        if audio_path is None:
            return ["Error: Could not prepare audio"]

        # Construct audio content using Baichuan format
        audio_content = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
        audio_content += ujson.dumps({'path': audio_path}, ensure_ascii=False)
        audio_content += tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)

        # Create messages in dictionary format (simpler approach)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": audio_content + "\n\n" + prompt},
        ]

        # Preprocess messages using Baichuan's text processing approach
        text = ""
        for msg in messages:
            text += role_prefix[msg['role']]
            text += msg['content']
        text += role_prefix["assistant"]
        
        # Process and generate
        pret = model.processor([text])
        plen = pret.input_ids.shape[1]
        
        # Generate text response
        outputs = model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(), 
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=tokenizer,
            max_new_tokens=10,  # Very restrictive
            do_sample=False,  # Deterministic
            temperature=0.0,
            return_dict_in_generate=True,
        )

        # Decode response
        response_text = tokenizer.decode(outputs.sequences[0, plen:])

        os.remove(audio_path)  # cleanup temp file
        return [response_text]

    except Exception as e:
        return [f"Error in Baichuan audio inference: {str(e)}"]

def inference_with_audio_and_text_clean(model, tokenizer, vocoder, audio_start_token, audio_end_token,
                                      audio_array, sr, transcription, prompt, sys_prompt="You are a helpful AI assistant."):
    """
    Clean inference function for audio+text inputs using Baichuan-Omni
    """
    try:
        # Prepare audio file
        audio_path = prepare_audio_for_baichuan(audio_array, sr)
        if audio_path is None:
            return ["Error: Could not prepare audio"]
        
        # Create audio content in Baichuan format
        audio_content = audio_start_token + ujson.dumps({'path': audio_path}, ensure_ascii=False) + audio_end_token
        
        # Combine transcription with prompt
        combined_prompt = f"Transcription: {transcription}\n\n{prompt}"
        
        # Create message content
        content = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": audio_content + "\n\n" + combined_prompt}
        ]
        
        # Preprocess messages
        text = ""
        for msg in content:
            text += role_prefix[msg['role']]
            text += msg['content']
        text += role_prefix["assistant"]
        
        # Process and generate
        pret = model.processor([text])
        plen = pret.input_ids.shape[1]
        
        # Generate text response
        textret = model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(), 
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=tokenizer,
            max_new_tokens=10,  # Very restrictive
            do_sample=False,  # Deterministic
            temperature=0.0,
            return_dict_in_generate=True,
        )
        
        # Decode response
        response_text = tokenizer.decode(textret.sequences[0, plen:])
        
        # Clean up temporary file
        try:
            os.remove(audio_path)
        except:
            pass
        
        return [response_text]
        
    except Exception as e:
        print(f"Error in Baichuan audio+text inference: {e}")
        return [f"Error: {str(e)}"]

def inference_with_text_clean(model, tokenizer, text_input, prompt, sys_prompt="You are a helpful AI assistant."):
    """
    Clean inference function for text-only inputs using Baichuan-Omni
    """
    try:
        # Combine transcription with prompt
        combined_prompt = f"Transcription: {text_input}\n\n{prompt}"
        
        # Create message content
        content = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": combined_prompt}
        ]
        
        # Preprocess messages
        text = ""
        for msg in content:
            text += role_prefix[msg['role']]
            text += msg['content']
        text += role_prefix["assistant"]
        
        # Process and generate
        pret = model.processor([text])
        plen = pret.input_ids.shape[1]
        
        # Generate text response
        textret = model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(), 
            audios=pret.audios.cuda() if pret.audios is not None else None,
            encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None else None,
            bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None else None,
            tokenizer=tokenizer,
            max_new_tokens=10,  # Very restrictive
            do_sample=False,  # Deterministic
            temperature=0.0,
            return_dict_in_generate=True,
        )
        
        # Decode response
        response_text = tokenizer.decode(textret.sequences[0, plen:])
        
        return [response_text]
        
    except Exception as e:
        print(f"Error in Baichuan text inference: {e}")
        return [f"Error: {str(e)}"]

def _normalize_label(label_str):
    value = label_str.lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value

def test_single_experiment(experiment_type, input_mode="audio", model_kind="baichuan", resume=True):
    """Test a single experiment type with checkpointing support
    
    Args:
        experiment_type: "1", "2A", "2B", "2C", "3A", "3B", "3C", "4"
        input_mode: "audio", "text", or "audio_and_text"
        model_kind: "baichuan" 
        resume: whether to resume from checkpoint if available
    """
    model_name = "Baichuan-Omni"
    print(f"Testing {model_name} on Experiment {experiment_type} ({input_mode})...")
    
    # Check system memory before proceeding
    import psutil
    memory_info = psutil.virtual_memory()
    print(f"System memory: {memory_info.available / (1024**3):.1f}GB available / {memory_info.total / (1024**3):.1f}GB total")
    
    # Setup checkpoint file
    model_prefix = "baichuan_omni" 
    checkpoint_file = get_checkpoint_filename(experiment_type, input_mode, model_prefix)
    
    # Load checkpoint if resuming
    results = []
    start_idx = 0
    experiment_info = {}
    
    if resume:
        results, last_processed_idx, experiment_info = load_checkpoint(checkpoint_file)
        start_idx = last_processed_idx + 1
        print(f"Resuming from sample {start_idx}")
    
    # Load dataset based on experiment type
    if experiment_type == "4":
        ds = load_dataset_with_schema_fix("VibeCheck1/LISTEN_full")  #  Correct
        # Use all samples from the paralinguistic dataset
        exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == "4")
    else:
        # Other experiments use VibeCheckAudio dataset - use non-streaming like Qwen3
        print("Loading dataset...")
        
        # Clear any existing cached data to free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Other experiments use VibeCheckAudio dataset
        ds = load_dataset_with_schema_fix("VibeCheck1/LISTEN_full")
        # Use all samples from the paralinguistic dataset


        # For experiments 2C and 3C, use samples from 2B and 3B respectively
        print(f"Filtering dataset for experiment {experiment_type}...")
        
        # Handle new experiment format (1_text, 1_audio, 1_audio_and_text -> 1)
        if experiment_type.startswith("1_"):
            actual_experiment_type = "1"
        elif experiment_type == "2C":
            actual_experiment_type = "2B"
        elif experiment_type == "3C":
            actual_experiment_type = "3B"
        else:
            actual_experiment_type = experiment_type
        
        exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == actual_experiment_type)
        
        # Convert filtered dataset to avoid dataclass issues
        try:
            exp_samples = exp_samples.to_iterable_dataset() if hasattr(exp_samples, 'to_iterable_dataset') else exp_samples
        except:
            pass  # If conversion fails, use as-is

    # NOW load the model after dataset filtering is complete
    try:
        num_samples = len(exp_samples)
        print(f"Dataset filtering complete! Found {num_samples} samples")
    except:
        print(f"Dataset filtering complete! Samples ready for processing")
    print("Loading model now that dataset filtering is done...")
    
    # Check memory again after filtering
    memory_info = psutil.virtual_memory()
    print(f"Memory after filtering: {memory_info.available / (1024**3):.1f}GB available")
    
    # Load model based on kind
    global model, tokenizer, vocoder, audio_start_token, audio_end_token
    if model_kind == "baichuan":
        try:
            model, tokenizer, vocoder, audio_start_token, audio_end_token = init_baichuan_model()
            print("Baichuan model loaded successfully")
        except Exception as e:
            print(f"Error: Failed to load Baichuan model: {e}")
            raise
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    
    # Check memory after model loading
    memory_info = psutil.virtual_memory()
    print(f"Memory after model loading: {memory_info.available / (1024**3):.1f}GB available")
    
    # Process all samples
    # Calculate total_samples safely to avoid dataclass issues
    try:
        total_samples = len(exp_samples)
        print(f'exp_samples: {total_samples}')
    except Exception as e:
        print(f'Warning: Could not get dataset length (dataclass issue): {e}')
        # Fallback: count manually by iterating (this will materialize the dataset)
        total_samples = sum(1 for _ in exp_samples)
        print(f'exp_samples (counted): {total_samples}')
        # Re-filter to get a fresh iterable
        if experiment_type == "4":
            ds = load_dataset_with_schema_fix("VibeCheck1/LISTEN_full")  #  Correct
            exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == "4")
        else:
            ds = load_dataset_with_schema_fix("VibeCheck1/LISTEN_full")  #  Correct
            if experiment_type.startswith("1_"):
                actual_experiment_type = "1"
            elif experiment_type == "2C":
                actual_experiment_type = "2B"
            elif experiment_type == "3C":
                actual_experiment_type = "3B"
            else:
                actual_experiment_type = experiment_type
            exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == actual_experiment_type)
    
    correct_predictions = len([r for r in results if r.get('correct', False)])
    print(f"\nProcessing {total_samples} samples from Experiment {experiment_type}...")
    if start_idx > 0:
        print(f"Already processed: {len(results)}/{total_samples} samples")
        print(f"Current accuracy: {correct_predictions}/{len(results)} ({correct_predictions/len(results)*100:.1f}%)" if results else "")
        print(f"Continuing from sample {start_idx}...")
    else:
        print("Starting fresh...")
    print("This may take a while...")
    
    # Initialize experiment info if not loaded from checkpoint
    if not experiment_info:
        actual_model_name = "BaiChuanInc/Baichuan-Omni-1.5" 
        experiment_info = {
            'model_name': actual_model_name,
            'experiment_type': experiment_type,
            'input_type': input_mode,
            'total_samples': total_samples,
            'start_time': datetime.now().isoformat()
        }
    
    for sample_idx in range(start_idx, total_samples):
        if sample_idx % 10 == 0:
            current_total = len(results)
            current_correct = len([r for r in results if r.get('correct', False)])
            print(f"\nProgress: {sample_idx}/{total_samples} ({sample_idx/total_samples*100:.1f}%)")
            if current_total > 0:
                print(f"Current accuracy: {current_correct}/{current_total} ({current_correct/current_total*100:.1f}%)")
        
        original_sample = exp_samples[sample_idx]
       
        # Clean the sample data
        cleaned_sample = clean_sample_data(original_sample)
        
        # Randomize choice order
        randomized_sample, expected_letter = randomize_choices(cleaned_sample, input_mode, original_sample, experiment_type)
        
        if expected_letter is None:
            # Check if this was due to empty answer (already warned) or actual error
            # For text mode experiments 2A/3A, check explicit_emotion; for all others, check answer
            if input_mode == "text" and experiment_type in ["2A", "3A"]:
                answer_field = original_sample.get('explicit_emotion', '')
            else:
                answer_field = original_sample.get('answer', '')
            
            if not answer_field or answer_field.strip() == '':
                # This was an empty answer case, already warned, just skip silently
                continue
            else:
                print(f"Error in randomization for sample {sample_idx}, skipping...")
                continue
        
        # Create clean prompt with randomized choices
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(randomized_sample['choices'])])
        
        # Create prompt based on input mode
        if input_mode == "audio":
            # Audio-based prompt for experiments 1, 2B, 3B
            prompt = f"""Listen to the audio and classify the emotion.

{randomized_sample['question']}

{choices_text}

Respond with only the letter (A, B, C, D, E, F, G, H):"""
        elif input_mode == "text":
            # Text-based prompt for experiments 2A, 3A
            prompt = f"""Read the transcription and classify the emotion.

{randomized_sample['question']}

{choices_text}

Respond with only the letter (A, B, C, D, E, F, G, H):"""
        elif input_mode == "audio_and_text":
            # Audio and text prompt for experiments 2C, 3C
            prompt = f"""Listen to the audio and read the transcription, then classify the emotion.

{randomized_sample['question']}

{choices_text}

Respond with only the letter (A, B, C, D, E, F, G, H):"""
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        
        # Store result data
        # Use explicit_emotion for text mode, answer for other modes
        ground_truth = original_sample.get('explicit_emotion', original_sample['answer']) if input_mode == "text" else original_sample['answer']
        
        result = {
            'sample_idx': original_sample['id'],
            'sample_id': cleaned_sample['id'],
            'transcription': original_sample['transcription'],
            'ground_truth': ground_truth,
            'original_choices': original_sample['choices'],
            'randomized_choices': randomized_sample['choices'],
            'expected_letter': expected_letter,
            'prompt': prompt,
            'input_type': input_mode,
            'correct': False,
            'predicted_letter': None,
            'predicted_emotion': None,
            'raw_response': None,
            'error': None
        }
        
        try:
            if input_mode == "audio":
                audio_data = original_sample['audio']
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                print(f"Audio array shape: {audio_array.shape}")
                
                # Handle HuggingFace AudioDecoder object - use default sampling rate
                sr = 16000  # Standard sampling rate for most audio datasets
                
                print("Step 4: Starting inference...")
                # Get prediction with audio
                response = inference_with_audio_clean(
                    model, tokenizer,
                    audio_array=audio_array,
                    sr=sr,
                    prompt=prompt,
                    sys_prompt="You are a helpful AI assistant capable of analyzing audio and text to classify emotions accurately."
                )
                
                if response and len(response) > 0 and not response[0].startswith("Error"):
                    full_response = response[0]
                    result['raw_response'] = full_response
                    
                    # Extract letter from response
                    predicted_letter = None
                    for char in full_response:
                        if char in 'ABCDEFGH':
                            predicted_letter = char
                            break
                    
                    result['predicted_letter'] = predicted_letter
                    
                    if predicted_letter == expected_letter:
                        correct_predictions += 1
                        result['correct'] = True
                    else:
                        # Show what the model actually predicted
                        if predicted_letter:
                            try:
                                pred_idx = ord(predicted_letter) - ord('A')
                                if 0 <= pred_idx < len(randomized_sample['choices']):
                                    predicted_emotion = randomized_sample['choices'][pred_idx]
                                    result['predicted_emotion'] = predicted_emotion
                            except:
                                pass
                else:
                    result['error'] = response[0] if response else "No valid response"
            
            elif input_mode == "audio_and_text":
                # For experiments 2C and 3C - use both audio and transcription
                audio_data = original_sample['audio']
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                
                # Handle HuggingFace AudioDecoder object - use default sampling rate
                sr = 16000  # Standard sampling rate for most audio datasets
                
                # Get transcription
                transcription = original_sample.get('transcription', '')
                if not transcription:
                    result['error'] = "No transcription available"
                    results.append(result)
                    continue
                
                print("Step 4: Starting inference with audio and text...")
                # Get prediction with both audio and transcription
                response = inference_with_audio_and_text_clean(
                    model, tokenizer, vocoder, audio_start_token, audio_end_token,
                    audio_array=audio_array,
                    sr=sr,
                    transcription=transcription,
                    prompt=prompt,
                    sys_prompt="You are a helpful AI assistant capable of analyzing audio and text to classify emotions accurately."
                )
                
                if response and len(response) > 0 and not response[0].startswith("Error"):
                    full_response = response[0]
                    result['raw_response'] = full_response
                    
                    # Extract letter from response
                    predicted_letter = None
                    for char in full_response:
                        if char in 'ABCDEFGH':
                            predicted_letter = char
                            break
                    
                    result['predicted_letter'] = predicted_letter
                    
                    if predicted_letter == expected_letter:
                        correct_predictions += 1
                        result['correct'] = True
                    else:
                        # Show what the model actually predicted
                        if predicted_letter:
                            try:
                                pred_idx = ord(predicted_letter) - ord('A')
                                if 0 <= pred_idx < len(randomized_sample['choices']):
                                    predicted_emotion = randomized_sample['choices'][pred_idx]
                                    result['predicted_emotion'] = predicted_emotion
                            except:
                                pass
                else:
                    result['error'] = response[0] if response else "No valid response"
            
            elif input_mode == "text":
                # Process text transcription
                transcription = original_sample.get('transcription', '')
                if not transcription:
                    result['error'] = "No transcription available"
                    results.append(result)
                    continue
                
                if model_kind == "baichuan":
                    # Get prediction with text using Baichuan
                    response = inference_with_text_clean(
                        model, tokenizer,
                        text_input=transcription,
                        prompt=prompt,
                        sys_prompt="You are a helpful AI assistant capable of analyzing text to classify emotions accurately."
                    )

                    if response and len(response) > 0 and not response[0].startswith("Error"):
                        full_response = response[0]
                        result['raw_response'] = full_response
                        
                        # Extract letter from response
                        predicted_letter = None
                        for char in full_response:
                            if char in 'ABCDEFGH':
                                predicted_letter = char
                                break
                        
                        result['predicted_letter'] = predicted_letter
                        
                        if predicted_letter == expected_letter:
                            correct_predictions += 1
                            result['correct'] = True
                        else:
                            # Show what the model actually predicted
                            if predicted_letter:
                                try:
                                    pred_idx = ord(predicted_letter) - ord('A')
                                    if 0 <= pred_idx < len(randomized_sample['choices']):
                                        predicted_emotion = randomized_sample['choices'][pred_idx]
                                        result['predicted_emotion'] = predicted_emotion
                                except:
                                    pass
                    else:
                        result['error'] = response[0] if response else "No valid response"
                
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
        
        # Save checkpoint every 10 samples or on errors
        if sample_idx % 10 == 0 or result.get('error'):
            try:
                save_checkpoint(checkpoint_file, results, sample_idx, experiment_info)
                # Clear GPU cache periodically to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
        
        # Note: correct_predictions is already updated when prediction is determined to be correct
    
    # Calculate final accuracy
    accuracy = (correct_predictions / len(results)) * 100 if results else 0
    
    # Update experiment info with final stats
    experiment_info.update({
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'completion_timestamp': datetime.now().isoformat(),
        'total_processed': len(results)
    })
    
    # Final checkpoint save
    try:
        save_checkpoint(checkpoint_file, results, total_samples - 1, experiment_info)
        print("Final checkpoint saved")
    except Exception as e:
        print(f"Warning: Failed to save final checkpoint: {e}")
    
    # Save final results to timestamped file
    results_data = {
        'experiment_info': experiment_info,
        'results': results
    }
    
    # Save to JSON file
    output_file = f"results/{model_prefix}_experiment{experiment_type}_{input_mode}_results_3allemo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Clean up checkpoint file on successful completion
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("Checkpoint file cleaned up")
    except Exception as e:
        print(f"Warning: Could not remove checkpoint file: {e}")
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment_type} RESULTS ({input_mode.upper()})")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}/{total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {output_file}")
    
    # Calculate comprehensive evaluation metrics
    try:
        comprehensive_metrics = calculate_comprehensive_metrics(results)
        print_comprehensive_results(comprehensive_metrics, experiment_type, input_mode, len(results))
        
        # Add comprehensive metrics to experiment info
        experiment_info.update({
            'comprehensive_metrics': comprehensive_metrics
        })
        
        # Update the saved results with comprehensive metrics
        results_data['experiment_info'] = experiment_info
        
        # Re-save with comprehensive metrics
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print(f"Results updated with comprehensive metrics")
        
    except Exception as e:
        print(f"Warning: Error calculating comprehensive metrics: {e}")
        print("Continuing with basic accuracy only...")
    
    if accuracy > 90:
        print("Warning: Still suspiciously high accuracy - check for other leakage sources")
    elif accuracy > 70:
        print("Good performance - model appears to be working correctly")
    else:
        print("Warning: Lower performance - might indicate the model is struggling without leaked information")
    
    return results_data

def test_all_experiments(model_kind="baichuan", resume=True):
    """Test all experiment types"""
    print("Testing models on Experiments WITHOUT data leakage...")
    
    # DO NOT load model here - will load after dataset filtering to save memory
    global model, tokenizer, vocoder, audio_start_token, audio_end_token
    print("Delaying model loading until after dataset filtering to optimize memory...")
    
    # Set memory fraction to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use only 85% of GPU memory
    
    # Define experiments to run
    if model_kind == "baichuan":
        experiments = [
            ("1_text", "text"),           # Experiment 1: Text only
            ("1_audio", "audio"),         # Experiment 1: Audio only
            ("1_audio_and_text", "audio_and_text"), # Experiment 1: Audio + Text
            ("2A", "text"),               # Experiment 2A: Text only
            ("2B", "audio"),              # Experiment 2B: Audio only
            ("2C", "audio_and_text"),     # Experiment 2C: Audio + Text
            ("3A", "text"),               # Experiment 3A: Text only
            ("3B", "audio"),              # Experiment 3B: Audio only
            ("3C", "audio_and_text"),     # Experiment 3C: Audio + Text
            ("4", "audio"),               # Experiment 4: Paralinguistic Audio only
        ]
    else:
        experiments = [
            ("2A", "text"),           # Experiment 2A: Text only
            ("3A", "text"),           # Experiment 3A: Text only
        ]
    
    all_results = {}
    
    for exp_type, input_mode in experiments:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT {exp_type} ({input_mode.upper()})")
        print(f"{'='*80}")
        
        try:
            results = test_single_experiment(exp_type, input_mode, model_kind=model_kind, resume=resume)
            
            all_results[f"experiment_{exp_type}"] = results
        except Exception as e:
            print(f"Error in experiment {exp_type}: {e}")
            all_results[f"experiment_{exp_type}"] = {"error": str(e)}
    
    # Save combined results
    model_prefix = "baichuan_omni"
    combined_output_file = f"results/{model_prefix}_all_experiments_results_3allemo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Combined results saved to: {combined_output_file}")
    
    # Print summary
    for exp_type, input_mode in experiments:
        if f"experiment_{exp_type}" in all_results and "error" not in all_results[f"experiment_{exp_type}"]:
            exp_info = all_results[f"experiment_{exp_type}"]["experiment_info"]
            print(f"Experiment {exp_type} ({input_mode}): {exp_info['accuracy']:.1f}% accuracy")
    
    return all_results

def run_selected_experiments(model_kind="baichuan", experiments_to_run=None, resume=True):
    """Run only selected experiments"""
    print("Testing models on selected experiments WITHOUT data leakage...")
    
    # DO NOT load model here - will load after dataset filtering to save memory
    global model, tokenizer, vocoder, audio_start_token, audio_end_token
    print("Delaying model loading until after dataset filtering to optimize memory...")
    
    # Set memory fraction to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # Use only 85% of GPU memory
    
    # Define all possible experiments
    all_experiments = {
        "1_text": ("1", "text"),
        "1_audio": ("1", "audio"),
        "1_audio_and_text": ("1", "audio_and_text"),
        "2A": ("2A", "text"),
        "2B": ("2B", "audio"),
        "2C": ("2C", "audio_and_text"),
        "3A": ("3A", "text"),
        "3B": ("3B", "audio"),
        "3C": ("3C", "audio_and_text"),
        "4": ("4", "audio"),
    }
    
    # Filter experiments based on model capabilities and user selection

    available_experiments = all_experiments
    
    # Use selected experiments or all available ones
    if experiments_to_run:
        # Validate selected experiments
        invalid_experiments = set(experiments_to_run) - set(available_experiments.keys())
        if invalid_experiments:
            print(f"Warning: Invalid experiments for {model_kind} model: {invalid_experiments}")
            print(f"Available experiments: {list(available_experiments.keys())}")
        
        # Filter to valid experiments
        experiments = [(exp_type, input_mode) for exp_key, (exp_type, input_mode) in available_experiments.items() 
                      if exp_key in experiments_to_run]
    else:
        experiments = list(available_experiments.values())
    
    if not experiments:
        print("No valid experiments to run!")
        return {}
    
    print(f"Will run experiments: {[exp[0] for exp in experiments]}")
    
    all_results = {}
    
    for exp_type, input_mode in experiments:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT {exp_type} ({input_mode.upper()})")
        print(f"{'='*80}")
        
        try:
            results = test_single_experiment(exp_type, input_mode, model_kind=model_kind, resume=resume)
            all_results[f"experiment_{exp_type}"] = results
        except Exception as e:
            print(f"Error in experiment {exp_type}: {e}")
            all_results[f"experiment_{exp_type}"] = {"error": str(e)}
    
    # Save combined results
    model_prefix = "baichuan_omni"
    exp_suffix = "_".join([exp[0] for exp in experiments]) if len(experiments) < len(available_experiments) else "all"
    combined_output_file = f"results/{model_prefix}_{exp_suffix}_experiments_results_3allemo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"SELECTED EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Combined results saved to: {combined_output_file}")
    
    # Print summary
    for exp_type, input_mode in experiments:
        if f"experiment_{exp_type}" in all_results and "error" not in all_results[f"experiment_{exp_type}"]:
            exp_info = all_results[f"experiment_{exp_type}"]["experiment_info"]
            print(f"Experiment {exp_type} ({input_mode}): {exp_info['accuracy']:.1f}% accuracy")
    
    return all_results

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    parser = argparse.ArgumentParser(description="Run VibeCheck experiments with selectable model and experiments")
    parser.add_argument("--model", type=str, default="baichuan", choices=["baichuan"], 
                       help="Model to use: baichuan (all experiments)")
    parser.add_argument("--experiments", type=str, nargs="+", 
                       choices=["1_text", "1_audio", "1_audio_and_text", "2A", "2B", "2C", "3A", "3B", "3C", "4"], 
                       help="Specific experiments to run (e.g., --experiments 1_text 1_audio 2A 3B 4). If not specified, runs all available experiments for the model.")
    parser.add_argument("--list-experiments", action="store_true", 
                       help="List all available experiments for the selected model and exit")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore any existing checkpoints")
    parser.add_argument("--list-checkpoints", action="store_true", help="List existing checkpoint files")
    parser.add_argument("--clean-checkpoints", action="store_true", help="Remove all checkpoint files")
    
    args = parser.parse_args()
    
    # Handle checkpoint management commands
    if args.list_checkpoints:
        print("\n Existing checkpoint files:")
        checkpoint_files = glob.glob("*_checkpoint.json")
        if checkpoint_files:
            for cf in sorted(checkpoint_files):
                try:
                    with open(cf, 'r') as f:
                        data = json.load(f)
                    print(f"  {cf}: {data.get('total_processed', 0)} samples, {data.get('timestamp', 'unknown time')}")
                except Exception as e:
                    print(f"  {cf}: (corrupted - {e})")
        else:
            print("  No checkpoint files found")
        print()
        exit(0)
    
    if args.clean_checkpoints:
        checkpoint_files = glob.glob("*_checkpoint.json")
        if checkpoint_files:
            print(f"\nRemoving {len(checkpoint_files)} checkpoint files...")
            for cf in checkpoint_files:
                try:
                    os.remove(cf)
                    print(f"  Removed: {cf}")
                except Exception as e:
                    print(f"  Failed to remove {cf}: {e}")
        else:
            print("\nNo checkpoint files to clean")
        print()
        exit(0)
    
    # List experiments if requested
    if args.list_experiments:
        print(f"\nAvailable experiments for {args.model} model:")
        if args.model == "baichuan":
            experiments_info = {
                "1_text": "Text-only emotion classification (using transcription)",
                "1_audio": "Audio-only emotion classification",
                "1_audio_and_text": "Audio + Text emotion classification",
                "2A": "Text-only emotion classification", 
                "2B": "Audio-only emotion classification",
                "2C": "Audio + Text emotion classification",
                "3A": "Text-only emotion classification",
                "3B": "Audio-only emotion classification", 
                "3C": "Audio + Text emotion classification",
                "4": "Paralinguistic Audio-only emotion classification"
            }
            for exp, desc in experiments_info.items():
                print(f"  {exp}: {desc}")
        print()
        exit(0)
    
    print("Testing with proper controls for data leakage")
    print("This test will:")
    print("1. Anonymize sample IDs")
    print("2. Remove metadata that could leak emotion")
    print("3. Randomize choice orders")
    
    if args.experiments:
        print(f"4. Test SELECTED experiments: {', '.join(args.experiments)}")
        if args.model == "baichuan":
            audio_exps = [e for e in args.experiments if e in ["1_audio", "2B", "3B", "4"]]
            text_exps = [e for e in args.experiments if e in ["1_text", "2A", "3A"]]
            audio_text_exps = [e for e in args.experiments if e in ["1_audio_and_text", "2C", "3C"]]
            if audio_exps:
                print(f"5. Use audio for experiments: {', '.join(audio_exps)}")
            if text_exps:
                print(f"6. Use text transcription for experiments: {', '.join(text_exps)}")
            if audio_text_exps:
                print(f"7. Use audio + text for experiments: {', '.join(audio_text_exps)}")
        else:
            print("5. Use text transcription for selected experiments")
    else:
        if args.model == "baichuan":
            print("4. Test ALL experiments: 1_text, 1_audio, 1_audio_and_text, 2A, 2B, 2C, 3A, 3B, 3C, 4")
            print("5. Use audio for experiments 1_audio, 2B, 3B, 4")
            print("6. Use text transcription for experiments 1_text, 2A, 3A")
            print("7. Use audio + text for experiments 1_audio_and_text, 2C, 3C")
        else:
            print("4. Test ONLY text experiments: 2A, 3A")
            print("5. Use text transcription for experiments 2A, 3A")
    print("8. Save detailed results to JSON files")
    print()
    
    # Determine resume setting
    resume = not args.no_resume
    if args.no_resume:
        print("Starting fresh (ignoring checkpoints)")
    else:
        print("Resume mode enabled (will continue from checkpoints if available)")
    
    # Run experiments
    if args.experiments:
        run_selected_experiments(model_kind=args.model, experiments_to_run=args.experiments, resume=resume)
    else:
        test_all_experiments(model_kind=args.model, resume=resume)
