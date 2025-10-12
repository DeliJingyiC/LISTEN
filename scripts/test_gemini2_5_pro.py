#!/usr/bin/env python3
"""
Gemini 2.5 pro test for VibeCheck experiments
Adapted from the Qwen2.5-Omni test script
"""

import pandas as pd
import json
import os
import time
from google import genai
from google.genai import types
from datasets import load_dataset
import numpy as np
import re
import random
import hashlib
import argparse
import tempfile
import wave
from io import BytesIO
from datetime import datetime
import glob
import shutil
from evaluation_utils import calculate_comprehensive_metrics, print_comprehensive_results

def setup_gemini_api():
    """Initialize Gemini API with API key"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            " No Gemini API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY"
        )

    client = genai.Client(vertexai=True,api_key=api_key)
    print(" Gemini API client configured successfully")
    return client

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
    """Load checkpoint if it exists; otherwise resume from latest results file.

    Fallback logic:
    - If checkpoint_file does not exist, attempt to locate the most recent
      results file for the same experiment and input type and resume from it.
    """
    # 1) Try loading the checkpoint file (standard path)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            prev_results = checkpoint_data.get('results', [])

            # Trim from the FIRST occurrence of 429 RESOURCE_EXHAUSTED onward
            first_quota_idx = None
            for i, r in enumerate(prev_results):
                err = r.get('error')
                if err and ("429" in str(err) and "RESOURCE_EXHAUSTED" in str(err)):
                    first_quota_idx = i
                    break

            if first_quota_idx is not None:
                print(
                    f" Trimming checkpoint results from first 429 at index {first_quota_idx} (removing {len(prev_results) - first_quota_idx} entries)"
                )
                prev_results = prev_results[:first_quota_idx]

            print(f" Found checkpoint with {len(prev_results)} non-429 processed samples")
            print(f" Last checkpoint: {checkpoint_data.get('timestamp', 'unknown')}")

            return (
                prev_results,
                len(prev_results) - 1,
                checkpoint_data.get('experiment_info', {})
            )
        except Exception as e:
            print(f" Error loading checkpoint: {e}")
            print("Starting fresh...")
            return [], -1, {}

    # 2) Fallback: Try resuming from the latest results file
    try:
        # Derive experiment and input type from the checkpoint filename
        base = os.path.basename(checkpoint_file)
        # Format: {prefix}_experiment{exp}_{input}_checkpoint.json
        import re
        m = re.match(r"(.*)_experiment([0-9A-Z]+)_(audio|text|audio_and_text)_checkpoint\.json", base)
        if m:
            prefix, exp, input_mode = m.groups()
            dirpath = os.path.dirname(checkpoint_file) or "."
            import glob
            pattern = os.path.join(dirpath, f"{prefix}_experiment{exp}_{input_mode}_results_*.json")
            candidates = sorted(glob.glob(pattern))
            if candidates:
                latest = candidates[-1]
                with open(latest, 'r') as f:
                    results_data = json.load(f)
                prev_results = results_data.get('results', [])
                experiment_info = results_data.get('experiment_info', {})

                # Trim from the FIRST occurrence of 429 RESOURCE_EXHAUSTED onward
                first_quota_idx = None
                for i, r in enumerate(prev_results):
                    err = r.get('error')
                    if err and ("429" in str(err) and "RESOURCE_EXHAUSTED" in str(err)):
                        first_quota_idx = i
                        break

                if first_quota_idx is not None:
                    print(
                        f" Trimming results from first 429 at index {first_quota_idx} (removing {len(prev_results) - first_quota_idx} entries)"
                    )
                    prev_results = prev_results[:first_quota_idx]

                resume_from = len(prev_results)
                print(f" No checkpoint found; resuming from latest results file: {os.path.basename(latest)}")
                print(f" Found {resume_from} non-429 samples in results; will resume from index {resume_from}")
                return prev_results, resume_from - 1, experiment_info
    except Exception as e:
        print(f" Error attempting results-based resumption: {e}")

    # 3) If neither checkpoint nor results are available
    return [], -1, {}

def get_checkpoint_filename(experiment_type, input_mode, model_prefix="gemini2_5_pro"):
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

def convert_audio_to_wav(audio_array, sample_rate=16000):
    """Convert numpy audio array to WAV format for Gemini API"""
    # Ensure audio is in the right format
    if audio_array.dtype != np.int16:
        # Convert float32 to int16
        if audio_array.dtype == np.float32:
            audio_array = (audio_array * 32767).astype(np.int16)
        else:
            audio_array = audio_array.astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

def prepare_audio_part(audio_data, sample_rate=16000):
    """Prepare audio data as a Part for the new Gemini API"""
    try:
        # Convert to WAV format
        wav_data = convert_audio_to_wav(audio_data, sample_rate)
        
        # Create Part from bytes
        audio_part = types.Part.from_bytes(
            data=wav_data,
            mime_type='audio/wav'
        )
        
        return audio_part
    except Exception as e:
        print(f"Error preparing audio: {e}")
        return None

def inference_with_audio_clean(client, audio_array, sr, prompt, sys_prompt="You are a helpful AI assistant capable of analyzing audio and text to classify emotions accurately."):
    """
    Clean inference function for audio-only inputs using Gemini 2.5 pro
    """
    try:
        # Prepare audio part
        audio_part = prepare_audio_part(audio_array, sr)
        if audio_part is None:
            return ["Error: Could not prepare audio"]
        
        # Generate response with audio and text prompt
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            # model='gemini-2.5-pro',
            contents=[
                prompt,
                audio_part
            ],
            # config=types.GenerateContentConfig(
            #     max_output_tokens=10,  # Very restrictive to match Qwen behavior
            #     temperature=0.0,  # Deterministic generation
            #     system_instruction=sys_prompt
            # )
        )
        print('response.text: ', response.text)
        return [response.text]
        
    except Exception as e:
        print(f"Error in audio inference: {e}")
        return [f"Error: {str(e)}"]

def inference_with_audio_and_text_clean(client, audio_array, sr, transcription, prompt, sys_prompt="You are a helpful AI assistant capable of analyzing audio and text to classify emotions accurately."):
    """
    Clean inference function for audio+text inputs using Gemini 2.5 pro
    """
    try:
        # Prepare audio part
        audio_part = prepare_audio_part(audio_array, sr)
        if audio_part is None:
            return ["Error: Could not prepare audio"]
        
        # Combine transcription with prompt
        combined_prompt = f"Transcription: {transcription}\n\n{prompt}"
        
        # Generate response with audio and text
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            # model='gemini-2.5-pro',
            contents=[
                combined_prompt,
                audio_part
            ]
        )
        
        return [response.text]
        
    except Exception as e:
        print(f"Error in audio+text inference: {e}")
        return [f"Error: {str(e)}"]

def inference_with_text_clean(client, text_input, prompt, sys_prompt="You are a helpful AI assistant capable of analyzing text to classify emotions accurately."):
    """
    Clean inference function for text-only inputs using Gemini 2.5 pro
    """
    try:
        # Combine transcription with prompt
        combined_prompt = f"Transcription: {text_input}\n\n{prompt}"
        
        # Generate response
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            # model='gemini-2.5-pro',
            contents=[combined_prompt],
        )
        
        return [response.text]
        
    except Exception as e:
        print(f"Error in text inference: {e}")
        return [f"Error: {str(e)}"]

def parse_resume_index_from_log(log_path, experiment_type, input_mode):
    """Parse a log file to find the next sample index to resume from for a given experiment.

    Heuristic:
    - Find the section header line for the exact experiment, which looks like
      "STARTING EXPERIMENT {exp} ({INPUT})".
    - Within that section, track the latest line of the form "Sample N:".
    - If a 429 quota error appears, assume all subsequent samples failed; resume
      from the last seen non-error sample index + 1.
    - If no sample lines are found, return 0.
    """
    try:
        in_section = False
        last_sample_idx = None
        saw_quota_error = False

        # Normalize input mode to match the banner (UPPERCASE)
        banner_input = input_mode.upper()

        start_banner = f"STARTING EXPERIMENT {experiment_type} ({banner_input})"

        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                if not in_section:
                    if start_banner in line:
                        in_section = True
                    continue

                # If we are in the section, capture latest sample index lines
                # Example: "Sample 123: 100/124 correct (80.6%)"
                m = re.search(r"Sample\s+(\d+):", line)
                if m:
                    try:
                        last_sample_idx = int(m.group(1))
                    except Exception:
                        pass

                # Detect 429 quota errors inside the section
                if "429" in line and "RESOURCE_EXHAUSTED" in line:
                    saw_quota_error = True
                    # We can break; we already have the last non-error seen
                    break

        if last_sample_idx is not None:
            # Resume from next index after the last recorded sample
            return last_sample_idx + 1
        return 0
    except Exception:
        return 0


def find_resume_log(directory_path, experiment_type, input_mode, model_prefix="gemini2_5_pro"):
    """Scan a directory for .log files that contain the experiment banner.

    Returns the log file with the most progress (highest sample number), or None.
    Only looks for logs from the specified model.
    """
    try:
        banner = f"STARTING EXPERIMENT {experiment_type} ({input_mode.upper()})"
        candidates = []
        for path in glob.glob(os.path.join(directory_path, "*.log")):
            # Only consider logs from the specified model
            if model_prefix not in os.path.basename(path):
                continue
                
            try:
                with open(path, "r", errors="ignore") as f:
                    in_section = False
                    last_sample_idx = None
                    
                    for line in f:
                        if not in_section:
                            if banner in line:
                                in_section = True
                            continue
                        
                        # Look for sample progress lines
                        m = re.search(r"Sample\s+(\d+):", line)
                        if m:
                            try:
                                last_sample_idx = int(m.group(1))
                            except Exception:
                                pass
                    
                    if last_sample_idx is not None:
                        candidates.append((last_sample_idx, path))
            except Exception:
                continue
        if not candidates:
            return None
        # Sort by sample number (highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    except Exception:
        return None


def test_single_experiment(client, experiment_type, input_mode="audio", resume=True, resume_from_log=None):
    """Test a single experiment type with checkpointing support
    
    Args:
        experiment_type: "1", "2A", "2B", "2C", "3A", "3B", "3C", "4"
        input_mode: "audio", "text", or "audio_and_text"
        resume: whether to resume from checkpoint if available
    """
    print(f"Testing Gemini 2.5 pro on Experiment {experiment_type} ({input_mode})...")
    
    # Setup checkpoint file
    checkpoint_file = get_checkpoint_filename(experiment_type, input_mode)
    
    # Load checkpoint if resuming
    results = []
    start_idx = 0
    experiment_info = {}
    
    # Prioritize checkpoints/results; only use logs if no prior progress exists
    if resume:
        results, last_processed_idx, experiment_info = load_checkpoint(checkpoint_file)
        start_idx = last_processed_idx + 1
        print(f" Resuming from saved progress: sample {start_idx}")
    
    if resume_from_log:
        log_resume_idx = parse_resume_index_from_log(resume_from_log, experiment_type, input_mode)
        if start_idx == 0 and log_resume_idx is not None and log_resume_idx > 0:
            print(f" Resume-from-log active (no prior progress) → start index: {log_resume_idx}")
            start_idx = log_resume_idx
        elif start_idx > 0:
            print(" Checkpoint/results progress found; ignoring resume-from-log to prioritize results")
    
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
    had_prior_results = len(results) > 0
    
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
        experiment_info = {
            'model_name': "Google Gemini 2.5 pro",
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
                    client=client,
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
                    client=client,
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
                
                # Get prediction with text using Gemini
                response = inference_with_text_clean(
                    client=client,
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
                # Use the actual number of results processed, not the loop index
                actual_processed_idx = len(results) - 1
                save_checkpoint(checkpoint_file, results, actual_processed_idx, experiment_info)
            except Exception as e:
                print(f" Failed to save checkpoint: {e}")
        
        # Note: correct_predictions is already updated when prediction is determined to be correct
        
        # Print progress for current sample (using recalculated count for safety)
        actual_correct = len([r for r in results if r.get('correct', False)])
        current_accuracy = (actual_correct / len(results)) * 100 if results else 0
        print(f"Sample {len(results)-1}: {actual_correct}/{len(results)} correct ({current_accuracy:.1f}%)")
        
        # Add small delay to respect API rate limits
        time.sleep(0.1)
    
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
        # Use the actual number of results processed, not the total samples
        actual_processed_idx = len(results) - 1
        save_checkpoint(checkpoint_file, results, actual_processed_idx, experiment_info)
        print(" Final checkpoint saved")
    except Exception as e:
        print(f" Failed to save final checkpoint: {e}")
    
    # Save final results to timestamped file
    results_data = {
        'experiment_info': experiment_info,
        'results': results
    }
    
    # Save to JSON file
    os.makedirs('results', exist_ok=True)
    if had_prior_results:
        results_data['experiment_info']['merged_from_previous'] = True
    output_file = f"results/gemini2_5_pro_experiment{experiment_type}_{input_mode}_results_3allemo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Retain checkpoint file for auditing/resumption purposes
    if os.path.exists(checkpoint_file):
        print(" Retaining checkpoint file for future resumption")
    
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

def test_all_experiments(resume=True, resume_from_log=None, auto_resume_from_logs=False, logs_dir=None):
    """Test all experiment types"""
    print("Testing Gemini 2.5 pro on Experiments WITHOUT data leakage...")
    
    # Initialize Gemini API
    client = setup_gemini_api()
    
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
            chosen_log = resume_from_log
            if not chosen_log and auto_resume_from_logs and logs_dir:
                candidate = find_resume_log(logs_dir, exp_type, input_mode, "gemini2_5_pro")
                if candidate:
                    print(f" Auto-selected resume log for {exp_type}({input_mode}): {os.path.basename(candidate)}")
                    chosen_log = candidate

            # Always prioritize log-based resume when available
            if chosen_log:
                print(f" Using log-based resume: {os.path.basename(chosen_log)}")
                results = test_single_experiment(client, exp_type, input_mode, resume=False, resume_from_log=chosen_log)
            else:
                results = test_single_experiment(client, exp_type, input_mode, resume=resume, resume_from_log=None)
            all_results[f"experiment_{exp_type}"] = results
        except Exception as e:
            print(f" Error in experiment {exp_type}: {e}")
            all_results[f"experiment_{exp_type}"] = {"error": str(e)}
    
    # Save combined results
    combined_output_file = f"gemini2_5_pro_all_experiments_results_3allemo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
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

def run_selected_experiments(experiments_to_run=None, resume=True, resume_from_log=None, force_input_mode=None, auto_resume_from_logs=False, logs_dir=None):
    """Run only selected experiments"""
    print("Testing Gemini 2.5 pro on selected experiments WITHOUT data leakage...")
    
    # Initialize Gemini API
    client = setup_gemini_api()
    
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
    
    # Use selected experiments or all available ones
    if experiments_to_run:
        # Validate selected experiments
        invalid_experiments = set(experiments_to_run) - set(all_experiments.keys())
        if invalid_experiments:
            print(f"Warning: Invalid experiments: {invalid_experiments}")
            print(f"Available experiments: {list(all_experiments.keys())}")
        
        # Filter to valid experiments
        experiments = [all_experiments[exp_type] for exp_type in experiments_to_run 
                      if exp_type in all_experiments]
    else:
        experiments = list(all_experiments.values())
    
    if not experiments:
        print("No valid experiments to run!")
        return {}
    
    if force_input_mode:
        fim = force_input_mode.lower()
        valid_modes = {"audio", "text", "audio_and_text"}
        if fim not in valid_modes:
            print(f"Warning: --force-input-mode must be one of {valid_modes}; ignoring '{force_input_mode}'")
        else:
            experiments = [(exp, fim) for exp, _ in experiments]
            print(f"Force input mode active → overriding all input modes to '{fim}'")

    print(f"Will run experiments: {[f'{exp[0]}({exp[1]})' for exp in experiments]}")
    
    all_results = {}
    
    for exp_type, input_mode in experiments:
        print(f"\n{'='*80}")
        print(f"STARTING EXPERIMENT {exp_type} ({input_mode.upper()})")
        print(f"{'='*80}")
        
        try:
            chosen_log = resume_from_log
            if not chosen_log and auto_resume_from_logs and logs_dir:
                candidate = find_resume_log(logs_dir, exp_type, input_mode, "gemini2_5_pro")
                if candidate:
                    print(f" Auto-selected resume log for {exp_type}({input_mode}): {os.path.basename(candidate)}")
                    chosen_log = candidate

            # Always prioritize log-based resume when available
            if chosen_log:
                print(f" Using log-based resume: {os.path.basename(chosen_log)}")
                results = test_single_experiment(client, exp_type, input_mode, resume=False, resume_from_log=chosen_log)
            else:
                results = test_single_experiment(client, exp_type, input_mode, resume=resume, resume_from_log=None)
            all_results[f"experiment_{exp_type}"] = results
        except Exception as e:
            print(f" Error in experiment {exp_type}: {e}")
            all_results[f"experiment_{exp_type}"] = {"error": str(e)}
    
    # Save combined results
    exp_suffix = "_".join([exp[0] for exp in experiments]) if len(experiments) < len(all_experiments) else "all"
    combined_output_file = f"results/gemini2_5_pro_{exp_suffix}_experiments_results_3allemo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    
    parser = argparse.ArgumentParser(description="Run VibeCheck experiments with Gemini 2.5 pro")
    parser.add_argument("--experiments", type=str, nargs="+", 
                       choices=["1_text", "1_audio", "1_audio_and_text", "2A", "2B", "2C", "3A", "3B", "3C", "4"], 
                       help="Specific experiments to run (e.g., --experiments 1_text 1_audio 2A 3B 4). If not specified, runs all experiments.")
    parser.add_argument("--list-experiments", action="store_true", 
                       help="List all available experiments and exit")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore any existing checkpoints")
    parser.add_argument("--list-checkpoints", action="store_true", help="List existing checkpoint files")
    parser.add_argument("--clean-checkpoints", action="store_true", help="Remove all checkpoint files")
    parser.add_argument("--resume-from-log", type=str, default=None, help="Path to a log file to infer resume index when checkpoints are missing")
    parser.add_argument("--force-input-mode", type=str, default=None, choices=["audio", "text", "audio_and_text"], help="Override input mode for all selected experiments (useful to match log section)")
    parser.add_argument("--auto-resume-from-logs", action="store_true", help="Auto-discover the newest matching log in --logs-dir and resume from it if --resume-from-log not provided")
    parser.add_argument("--logs-dir", type=str, default="/users/PAS2062/delijingyic/project/VibeCheckBench/src/output", help="Directory to scan for logs when using --auto-resume-from-logs")
    
    args = parser.parse_args()
    
    # Handle checkpoint management commands
    if args.list_checkpoints:
        print(" Existing checkpoint files:")
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
            print("\n No checkpoint files to clean")
        print()
        exit(0)
    
    # List experiments if requested
    if args.list_experiments:
        print(f"\nAvailable experiments for Gemini 2.5 pro:")
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
    
    print("Testing Gemini 2.5 pro with proper controls for data leakage")
    print("This test will:")
    print("1. Anonymize sample IDs")
    print("2. Remove metadata that could leak emotion")
    print("3. Randomize choice orders")
    
    if args.experiments:
        print(f"4. Test SELECTED experiments: {', '.join(args.experiments)}")
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
        run_selected_experiments(
            experiments_to_run=args.experiments,
            resume=resume,
            resume_from_log=args.resume_from_log,
            force_input_mode=args.force_input_mode,
            auto_resume_from_logs=args.auto_resume_from_logs,
            logs_dir=args.logs_dir,
        )
    else:
        # For all experiments, we ignore force_input_mode because suites fix input per design
        test_all_experiments(
            resume=resume,
            resume_from_log=args.resume_from_log,
            auto_resume_from_logs=args.auto_resume_from_logs,
            logs_dir=args.logs_dir,
        )
