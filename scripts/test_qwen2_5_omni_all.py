#!/usr/bin/env python3
"""
Fixed Qwen2.5-Omni test without data leakage
Added experiments 2C and 3C with both audio and transcription
"""

import pandas as pd
import json
import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from datasets import load_dataset
import numpy as np
import re
import random
import hashlib
import argparse
from datetime import datetime
import glob
import shutil

# Import the required utility function
from qwen_omni_utils import process_mm_info
from evaluation_utils import calculate_comprehensive_metrics, print_comprehensive_results

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
    print(f" Checkpoint saved: {len(results)} samples processed")

def load_checkpoint(checkpoint_file):
    """Load checkpoint if it exists"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f" Found checkpoint: {checkpoint_data['total_processed']} samples already processed")
            print(f" Last checkpoint: {checkpoint_data['timestamp']}")
            
            return (
                checkpoint_data.get('results', []),
                checkpoint_data.get('last_processed_idx', -1),
                checkpoint_data.get('experiment_info', {})
            )
        except Exception as e:
            print(f" Error loading checkpoint: {e}")
            print("Starting fresh...")
            return [], -1, {}
    
    return [], -1, {}

def get_checkpoint_filename(experiment_type, input_mode, model_prefix="qwen2_5_omni"):
    """Generate consistent checkpoint filename"""
    return f"results/{model_prefix}_experiment{experiment_type}_{input_mode}_checkpoint_3allemo.json"

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

def inference_with_audio_clean(audio_array, sr, prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."):
    """
    Clean inference function that doesn't leak information
    """
    
    # Create messages with NO identifying information
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user", 
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process multimedia info
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    print("Finished processing multimedia info")
    
    # Process inputs
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=True
    )
    inputs = inputs.to(model.device)
    print("model.device: ", model.device)
    print("Finished processing inputs")
    
    # Generate with controlled parameters
    output = model.generate(
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

def inference_with_audio_and_text_clean(audio_array, sr, transcription, prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."):
    """
    Clean inference function for experiments 2C and 3C with both audio and transcription
    """
    
    # Create messages with both audio and transcription
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user", 
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": f"Transcription: {transcription}\n\n{prompt}"},
            ]
        },
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process multimedia info
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    print("Finished processing multimedia info")
    
    # Process inputs
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=True
    )
    inputs = inputs.to(model.device)
    print("model.device: ", model.device)
    print("Finished processing inputs")
    
    # Generate with controlled parameters
    output = model.generate(
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

def inference_with_text_clean(text_input, prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."):
    """
    Clean inference function for text-based inputs (experiments 2A, 3A)
    """
    
    # Create messages with text input
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": f"Transcription: {text_input}\n\n{prompt}"},
            ]
        },
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process multimedia info (no audio/video for text-only)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # Process inputs
    inputs = processor(
        text=text, 
        audio=audios, 
        images=images, 
        videos=videos, 
        return_tensors="pt", 
        padding=True, 
        use_audio_in_video=False
    )
    inputs = inputs.to(model.device)
    
    # Generate with controlled parameters
    output = model.generate(
        **inputs, 
        use_audio_in_video=False, 
        return_audio=False,
        max_new_tokens=3,  # Very restrictive
    )
    
    # Decode output
    text_output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text_output

def _normalize_label(label_str):
    value = label_str.lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value

def test_single_experiment(experiment_type, input_mode="audio", model_kind="qwen", resume=True):
    """Test a single experiment type with checkpointing support
    
    Args:
        experiment_type: "1", "2A", "2B", "2C", "3A", "3B", "3C", "4"
        input_mode: "audio", "text", or "audio_and_text"
        model_kind: "qwen"
        resume: whether to resume from checkpoint if available
    """
    model_name = "Qwen2.5-Omni"
    print(f"Testing {model_name} on Experiment {experiment_type} ({input_mode})...")
    
    # Setup checkpoint file
    model_prefix = "qwen2_5_omni"
    checkpoint_file = get_checkpoint_filename(experiment_type, input_mode, model_prefix)
    
    # Load checkpoint if resuming
    results = []
    start_idx = 0
    experiment_info = {}
    
    if resume:
        results, last_processed_idx, experiment_info = load_checkpoint(checkpoint_file)
        start_idx = last_processed_idx + 1
        print(f" Resuming from sample {start_idx}")
    
    # Load dataset based on experiment type
    if experiment_type == "4":
        # Experiment 4 uses VibeCheckParalinguistic dataset
        # ds = load_dataset("VibeCheck1/VibeCheckParalinguistic")
        ds = load_dataset("VibeCheck1/LISTEN_full")
        # Use all samples from the paralinguistic dataset
        exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == "4")
    else:
        # Other experiments use VibeCheckAudio dataset
        ds = load_dataset("VibeCheck1/LISTEN_full")
        
        # For experiments 2C and 3C, use samples from 2B and 3B respectively
        if experiment_type == "2C":
            exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == "2B")
        elif experiment_type == "3C":
            exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == "3B")
        else:
            # Handle new experiment format (1_text, 1_audio, 1_audio_and_text -> 1)
            if experiment_type.startswith("1_"):
                actual_experiment_type = "1"
            else:
                actual_experiment_type = experiment_type
            
            exp_samples = ds['train'].filter(lambda x: x['experiment_type'] == actual_experiment_type)
    
    # Process all samples
    total_samples = len(exp_samples)
    correct_predictions = len([r for r in results if r.get('correct', False)])
    
    print(f"\nProcessing {total_samples} samples from Experiment {experiment_type}...")
    if start_idx > 0:
        print(f" Already processed: {len(results)}/{total_samples} samples")
        print(f" Current accuracy: {correct_predictions}/{len(results)} ({correct_predictions/len(results)*100:.1f}%)" if results else "")
        print(f"  Continuing from sample {start_idx}...")
    else:
        print(" Starting fresh...")
    print("This may take a while...")
    
    # Initialize experiment info if not loaded from checkpoint
    if not experiment_info:
        actual_model_name = "Qwen/Qwen2.5-Omni-7B"
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
                    audio_array=audio_array,
                    sr=sr,
                    prompt=prompt,
                    sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                )
                
                if response and len(response) > 0:
                    full_response = response[0]
                    result['raw_response'] = full_response
                    # Extract assistant response
                    if "assistant\n" in full_response:
                        assistant_response = full_response.split("assistant\n")[-1].strip()
                    else:
                        assistant_response = full_response.strip()
                    
                    # Extract letter
                    predicted_letter = None
                    for char in assistant_response:
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
                    result['error'] = "No valid response"
            
            elif input_mode == "audio_and_text":
                # For experiments 2C and 3C - use both audio and transcription
                audio_data = original_sample['audio']
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                # print(f"Audio array shape: {audio_array.shape}")
                
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
                    audio_array=audio_array,
                    sr=sr,
                    transcription=transcription,
                    prompt=prompt,
                    sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                )
                
                if response and len(response) > 0:
                    full_response = response[0]
                    result['raw_response'] = full_response
                    # Extract assistant response
                    if "assistant\n" in full_response:
                        assistant_response = full_response.split("assistant\n")[-1].strip()
                    else:
                        assistant_response = full_response.strip()
                    
                    # Extract letter
                    predicted_letter = None
                    for char in assistant_response:
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
                    result['error'] = "No valid response"
            
            elif input_mode == "text":
                # Process text transcription
                transcription = original_sample.get('transcription', '')
                if not transcription:
                    result['error'] = "No transcription available"
                    results.append(result)
                    continue
                
                if model_kind == "qwen":
                    # Get prediction with text using Qwen
                    response = inference_with_text_clean(
                        text_input=transcription,
                        prompt=prompt,
                        sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                    )

                    if response and len(response) > 0:
                        full_response = response[0]
                        result['raw_response'] = full_response
                        # Extract assistant response
                        if "assistant\n" in full_response:
                            assistant_response = full_response.split("assistant\n")[-1].strip()
                        else:
                            assistant_response = full_response.strip()
                        
                        # Extract letter
                        predicted_letter = None
                        for char in assistant_response:
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
                        result['error'] = "No valid response"
                else:
                    result['error'] = f"Unsupported model_kind: {model_kind}"
                
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
        
        # Save checkpoint every 10 samples or on errors
        if sample_idx % 10 == 0 or result.get('error'):
            try:
                save_checkpoint(checkpoint_file, results, sample_idx, experiment_info)
            except Exception as e:
                print(f" Failed to save checkpoint: {e}")
        
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
        print(" Final checkpoint saved")
    except Exception as e:
        print(f" Failed to save final checkpoint: {e}")
    
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
            print("🧹 Checkpoint file cleaned up")
    except Exception as e:
        print(f" Could not remove checkpoint file: {e}")
    
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
        print(f" Results updated with comprehensive metrics")
        
    except Exception as e:
        print(f" Error calculating comprehensive metrics: {e}")
        print("Continuing with basic accuracy only...")
    
    if accuracy > 90:
        print(" Still suspiciously high accuracy - check for other leakage sources")
    elif accuracy > 70:
        print(" Good performance - model appears to be working correctly")
    else:
        print(" Lower performance - might indicate the model is struggling without leaked information")
    
    return results_data

def test_all_experiments(model_kind="qwen", resume=True):
    """Test all experiment types"""
    print("Testing models on Experiments WITHOUT data leakage...")
    
    # Load model once
    global model, processor
    
    if model_kind == "qwen":
        model_name = "Qwen/Qwen2.5-Omni-7B"
        print("Loading Qwen2.5-Omni model...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
           model_name,
           torch_dtype=torch.bfloat16,  # Use bfloat16 instead of "auto"
           device_map="auto",
           attn_implementation="flash_attention_2",
       )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    
    # Define experiments to run
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
    
    all_results = {}
    
    for exp_type, input_mode in experiments:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT {exp_type} ({input_mode.upper()})")
        print(f"{'='*80}")
        
        try:
            results = test_single_experiment(exp_type, input_mode, model_kind=model_kind, resume=resume)
            all_results[f"experiment_{exp_type}"] = results
        except Exception as e:
            print(f" Error in experiment {exp_type}: {e}")
            all_results[f"experiment_{exp_type}"] = {"error": str(e)}
    
    # Save combined results
    model_prefix = "qwen2_5_omni"
    combined_output_file = f"results/{model_prefix}_all_experiments_results_3allemo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
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

def run_selected_experiments(model_kind="qwen", experiments_to_run=None, resume=True):
    """Run only selected experiments"""
    print("Testing models on selected experiments WITHOUT data leakage...")
    
    # Load model once
    global model, processor
    
    if model_kind == "qwen":
        model_name = "Qwen/Qwen2.5-Omni-7B"
        print("Loading Qwen2.5-Omni model...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
           model_name,
           torch_dtype=torch.bfloat16,  # Use bfloat16 instead of "auto"
           device_map="auto",
           attn_implementation="flash_attention_2",
       )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    
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
            print(f" Error in experiment {exp_type}: {e}")
            all_results[f"experiment_{exp_type}"] = {"error": str(e)}
    
    # Save combined results
    model_prefix = "qwen2_5_omni"
    exp_suffix = "_".join([exp[0] for exp in experiments]) if len(experiments) < len(available_experiments) else "all"
    combined_output_file = f"results/{model_prefix}_{exp_suffix}_experiments_results_3allemo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    parser.add_argument("--model", type=str, default="qwen", choices=["qwen"], 
                       help="Model to use: qwen (all experiments)")
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
            print(f"\n Removing {len(checkpoint_files)} checkpoint files...")
            for cf in checkpoint_files:
                try:
                    os.remove(cf)
                    print(f"  Removed: {cf}")
                except Exception as e:
                    print(f"  Failed to remove {cf}: {e}")
        else:
            print("\n✨ No checkpoint files to clean")
        print()
        exit(0)
    
    # List experiments if requested
    if args.list_experiments:
        print(f"\nAvailable experiments for {args.model} model:")
        if args.model == "qwen":
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
        if args.model == "qwen":
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
        print("4. Test ALL experiments: 1_text, 1_audio, 1_audio_and_text, 2A, 2B, 2C, 3A, 3B, 3C, 4")
        print("5. Use audio for experiments 1_audio, 2B, 3B, 4")
        print("6. Use text transcription for experiments 1_text, 2A, 3A")
        print("7. Use audio + text for experiments 1_audio_and_text, 2C, 3C")
    print("8. Save detailed results to JSON files")
    print()
    
    # Determine resume setting
    resume = not args.no_resume
    if args.no_resume:
        print(" Starting fresh (ignoring checkpoints)")
    else:
        print(" Resume mode enabled (will continue from checkpoints if available)")
    
    # Run experiments
    if args.experiments:
        run_selected_experiments(model_kind=args.model, experiments_to_run=args.experiments, resume=resume)
    else:
        test_all_experiments(model_kind=args.model, resume=resume)