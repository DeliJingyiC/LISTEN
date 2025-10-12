#!/usr/bin/env python3
"""
Qwen3-Instruct test without data leakage
Modified to use Qwen3-30B-A3B-Instruct-2507 for text-only experiments
"""

import pandas as pd
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import re
import random
import hashlib
import argparse
from datetime import datetime
import glob
import shutil
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

def get_checkpoint_filename(experiment_type, input_mode, model_prefix="qwen3_instruct"):
    """Generate consistent checkpoint filename"""
    return f"results/{model_prefix}_experiment{experiment_type}_{input_mode}_checkpoint.json"

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

def randomize_choices(sample,original_sample, experiment_type=None):
    """Randomize the order of choices to prevent position bias"""
    choices = sample['choices'].copy()
    
    # Determine correct answer based on experiment type
    if experiment_type == "3A":
        # For experiment 3A, use explicit_emotion
        correct_answer = original_sample.get('explicit_emotion', '')
    else:
        # For experiment 1_text and 2A, use answer field
        correct_answer = sample['answer']
    
    # Skip samples with empty answers
    if not correct_answer or correct_answer.strip() == '':
        print(f"Warning: Empty answer found, skipping randomization for this sample")
        return sample, None
    
    # Find original position
    try:
        original_position = choices.index(correct_answer)
    except ValueError:
        print(f"Error: Answer '{correct_answer}' not found in choices {choices}")
        return sample, None
    
    # Create randomized mapping
    indices = list(range(len(choices)))
    random.shuffle(indices)
    
    # Create new choices in randomized order
    randomized_choices = [choices[i] for i in indices]
    
    # Find new position of correct answer
    new_position = indices.index(original_position)
    new_letter = chr(65 + new_position)
    
    # Create modified sample
    randomized_sample = sample.copy()
    randomized_sample['choices'] = randomized_choices
    
    return randomized_sample, new_letter

def inference_with_text_clean(text_input, prompt, sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of generating text."):
    """
    Clean inference function for text-based inputs using Qwen3-Instruct
    """
    
    # Create messages with text input
    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user", 
            "content": f"Transcription: {text_input}\n\n{prompt}"
        },
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Process inputs
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate with controlled parameters
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=3,
        do_sample=False,  # Deterministic generation
        num_beams=1,      # No beam search
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Extract only the new tokens
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    text_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    return [text_output]


def _normalize_label(label_str):
    value = label_str.lower().strip()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value

def test_single_experiment(experiment_type, input_mode="text", model_kind="qwen3", resume=True):
    """Test a single experiment type with checkpointing support
    
    Args:
        experiment_type: "1", "2A", "3A" (Qwen3-Instruct only supports text)
        input_mode: "text" (Qwen3-Instruct is text-only)
        model_kind: "qwen3"
        resume: whether to resume from checkpoint if available
    """
    model_name = "Qwen3-Instruct"
    print(f"Testing {model_name} on Experiment {experiment_type} ({input_mode})...")
    
    # Setup checkpoint file
    model_prefix = "qwen3_instruct"
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
    # Qwen3-Instruct is text-only, so all experiments use VibeCheckAudio with transcription
    ds = load_dataset("VibeCheck1/LISTEN_full")
    
    # Handle new experiment format (1_text -> 1)
    if experiment_type == "1_text":
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
        actual_model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
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
        randomized_sample, expected_letter = randomize_choices(cleaned_sample,original_sample, experiment_type)
        
        if expected_letter is None:
            # Check if this was due to empty answer (already warned) or actual error
            # For experiments 2A/3A, check explicit_emotion; for experiment 1_text, check answer
            if experiment_type in ["3A"]:
                answer_field = original_sample.get('explicit_emotion', '')
            else:  # experiment 1_text
                answer_field = original_sample.get('answer', '')
            
            if not answer_field or answer_field.strip() == '':
                # This was an empty answer case, already warned, just skip silently
                continue
            else:
                print(f"Error in randomization for sample {sample_idx}, skipping...")
                continue
        
        # Create clean prompt with randomized choices
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(randomized_sample['choices'])])
        
        # Create prompt for text-based input (Qwen3-Instruct is text-only)
        prompt = f"""Read the transcription and classify the emotion.

{randomized_sample['question']}

{choices_text}

Respond with only the letter (A, B, C, D, E, F, G, H):"""
        
        # Store result data
        result = {
            'sample_idx': original_sample['id'],
            'sample_id': cleaned_sample['id'],
            'transcription': original_sample['transcription'],
            'ground_truth': original_sample['answer'],
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
            # Process text transcription (Qwen3-Instruct is text-only)
            transcription = original_sample.get('transcription', '')
            if not transcription:
                result['error'] = "No transcription available"
                results.append(result)
                continue
            
            if model_kind == "qwen3":
                # Get prediction with text using Qwen3-Instruct
                response = inference_with_text_clean(
                    text_input=transcription,
                    prompt=prompt,
                    sys_prompt="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of generating text."
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
    output_file = f"results/{model_prefix}_experiment{experiment_type}_{input_mode}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Clean up checkpoint file on successful completion
    try:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(" Checkpoint file cleaned up")
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

def test_all_experiments(model_kind="qwen3", resume=True):
    """Test all experiment types"""
    print("Testing models on Experiments WITHOUT data leakage...")
    
    # Load model once
    global model, tokenizer
    
    if model_kind == "qwen3":
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        print("Loading Qwen3-Instruct model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    
    # Define experiments to run (Qwen3-Instruct only supports text)
    experiments = [
            ("1_text", "text"),          # Experiment 1: Text only (using transcription)
            ("2A", "text"),              # Experiment 2A: Text only
            ("3A", "text"),              # Experiment 3A: Text only
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
    model_prefix = "qwen3_instruct"
    combined_output_file = f"results/{model_prefix}_all_experiments_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
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

def run_selected_experiments(model_kind="qwen3", experiments_to_run=None, resume=True):
    """Run only selected experiments"""
    print("Testing models on selected experiments WITHOUT data leakage...")
    
    # Load model once
    global model, tokenizer
    
    if model_kind == "qwen3":
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        print("Loading Qwen3-Instruct model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    
    # Define all possible experiments (Qwen3-Instruct only supports text)
    all_experiments = {
        "1_text": ("1", "text"),
        "2A": ("2A", "text"),
        "3A": ("3A", "text"),
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
    model_prefix = "qwen3_instruct"
    exp_suffix = "_".join([exp[0] for exp in experiments]) if len(experiments) < len(available_experiments) else "all"
    combined_output_file = f"results/{model_prefix}_{exp_suffix}_experiments_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    
    parser = argparse.ArgumentParser(description="Run VibeCheck experiments with Qwen3-Instruct")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3"], 
                       help="Model to use: qwen3 (experiments 1_text,2A,3A)")
    parser.add_argument("--experiments", type=str, nargs="+", 
                       choices=["1_text", "2A", "3A"], 
                       help="Specific experiments to run (e.g., --experiments 1_text 2A 3A). If not specified, runs all available experiments for the model.")
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
            print("\n No checkpoint files to clean")
        print()
        exit(0)
    
    # List experiments if requested
    if args.list_experiments:
        print(f"\nAvailable experiments for {args.model} model:")
        if args.model == "qwen3":
            experiments_info = {
                "1_text": "Text-only emotion classification (using transcription)",
                "2A": "Text-only emotion classification", 
                "3A": "Text-only emotion classification"
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
        print("5. Use text transcription for all experiments (Qwen3-Instruct is text-only)")
    else:
        print("4. Test ALL experiments: 1_text, 2A, 3A")
        print("5. Use text transcription for all experiments (Qwen3-Instruct is text-only)")
    print("6. Save detailed results to JSON files")
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