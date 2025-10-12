"""
LISTEN Benchmark - Template for Testing Your Own Model

================================================================================
MINIMAL SETUP - Only 3 Steps Required!
================================================================================

STEP 1: Set your HuggingFace token (line 27)
STEP 2: Implement two inference functions (lines 40-92):
        - inference_with_audio_clean(): for audio inputs
        - inference_with_text_clean(): for text inputs
STEP 3: Run: python test_your_model.py --model "YourModel" --experiments 2A 3A

That's it! The script handles everything else automatically.

================================================================================
What you need to implement:
- inference_with_audio_clean(audio_array, sr, prompt): Process audio, return letter
- inference_with_text_clean(text, prompt): Process text, return letter

The script automatically handles:
- Data loading, preprocessing, randomization
- Checkpointing, metrics calculation, results saving
- Calling the right function based on experiment type
================================================================================
"""

import json, os, torch, numpy as np, random, hashlib, argparse
from datetime import datetime
from datasets import load_dataset
from evaluation_utils import calculate_comprehensive_metrics, print_comprehensive_results

# STEP 1: Set your HuggingFace token
os.environ['HUGGINGFACE_HUB_TOKEN'] = 'your_token_here'

# Global variables for your model
model = None
processor = None  # or tokenizer, or whatever you need

# ============================================================================
# STEP 2: IMPLEMENT THESE FUNCTIONS FOR YOUR MODEL
# ============================================================================

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
    
def load_your_model():
    """
    Optional: Implement this if you need to load a model.
    If your model is API-based (OpenAI, Anthropic), you can leave this empty.
    """
    global model, processor
    # TODO: Load your model here if needed
    # model = YourModel.from_pretrained("model_name")
    # processor = YourProcessor.from_pretrained("model_name")
    pass
# ============================================================================
# Everything below is automatic - no need to modify
# ============================================================================



def save_checkpoint(file, results, idx, info):
    data = {'results': results, 'last_idx': idx, 'info': info, 'time': datetime.now().isoformat()}
    os.makedirs(os.path.dirname(file), exist_ok=True)
    temp = file + '.tmp'
    with open(temp, 'w') as f: json.dump(data, f, default=str)
    os.rename(temp, file)

def load_checkpoint(file):
    if os.path.exists(file):
        try:
            with open(file) as f: data = json.load(f)
            print(f"Resuming: {data.get('last_idx', 0)+1} samples done")
            return data.get('results', []), data.get('last_idx', -1), data.get('info', {})
        except: pass
    return [], -1, {}

def randomize_choices(sample, original, exp_type):
    choices = sample['choices'].copy()
    
    # Get correct answer
    if exp_type == "3A":
        answer = original.get('explicit_emotion', sample.get('answer', ''))
    else:
        answer = sample.get('answer', '')
    
    if not answer or not answer.strip(): return sample, None
    
    # Add extra emotions for exp 3
    if exp_type in ["3A", "3B", "3C"]:
        choices = list(set(choices + ["neutral", "sadness", "surprise", "happiness", "fear", "anger", "excitement", "frustration"]))
    
    try:
        orig_pos = choices.index(answer)
        indices = list(range(len(choices)))
        random.shuffle(indices)
        new_choices = [choices[i] for i in indices]
        new_pos = indices.index(orig_pos)
        sample_copy = sample.copy()
        sample_copy['choices'] = new_choices
        return sample_copy, chr(65 + new_pos)
    except ValueError:
        return sample, None

def test_experiment(exp_type, input_mode, model_name="YourModel", resume=True):
    """Run one experiment"""
    print(f"\n{'='*80}\nTesting {model_name} on Experiment {exp_type} ({input_mode})\n{'='*80}")
    
    checkpoint = f"results/{model_name.lower().replace(' ','_')}_exp{exp_type}_{input_mode}_checkpoint.json"
    results, start_idx, info = load_checkpoint(checkpoint) if resume else ([], -1, {})
    start_idx += 1
    
    # Load data
    ds = load_dataset("VibeCheck1/LISTEN_full")
    if exp_type == "4":
        samples = ds['train'].filter(lambda x: x['experiment_type'] == "4")
    elif exp_type in ["2C", "3C"]:
        base_type = "2B" if exp_type == "2C" else "3B"
        samples = ds['train'].filter(lambda x: x['experiment_type'] == base_type)
    else:
        actual_type = "1" if exp_type.startswith("1_") else exp_type
        samples = ds['train'].filter(lambda x: x['experiment_type'] == actual_type)
    
    total = len(samples)
    if not info:
        info = {'model': model_name, 'exp': exp_type, 'mode': input_mode, 'total': total, 'start': datetime.now().isoformat()}
    
    print(f"Processing {total} samples (starting from {start_idx})...")
    
    for idx in range(start_idx, total):
        if idx % 10 == 0:
            acc = len([r for r in results if r.get('correct', False)]) / len(results) * 100 if results else 0
            print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%) - Accuracy: {acc:.1f}%")
        
        original = samples[idx]
        
        # Anonymize and randomize
        sample = {k: original[k] for k in ['question', 'choices', 'audio', 'answer', 'transcription'] if k in original}
        sample['id'] = hashlib.md5(original['id'].encode()).hexdigest()[:8]
        
        randomized, expected = randomize_choices(sample, original, exp_type)
        if not expected: continue
        
        # Create prompt
        choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(randomized['choices'])])
        prompt = f"{randomized['question']}\n\n{choices_text}\n\nRespond with only the letter (A, B, C, etc.):"
        
        result = {
            'sample_id': sample['id'],
            'ground_truth': original.get('explicit_emotion') if input_mode == "text" and exp_type == "3A" else original['answer'],
            'expected_letter': expected,
            'correct': False,
            'predicted_letter': None,
            'error': None
        }
        
        try:
            # Call your inference functions
            if input_mode == "audio" or input_mode == "audio_and_text":
                audio = np.array(original['audio']['array'], dtype=np.float32)
                response = inference_with_audio_clean(audio, sr=16000, prompt=prompt)
            elif input_mode == "text":
                text = original.get('transcription', '')
                response = inference_with_text_clean(text, prompt=prompt)
            
            # Extract letter from response
            predicted = None
            if isinstance(response, str):
                for char in response.upper():
                    if char in 'ABCDEFGH':
                        predicted = char
                        break
            
            result['predicted_letter'] = predicted
            if predicted == expected:
                result['correct'] = True
        
        except Exception as e:
            result['error'] = str(e)
            print(f"Error on sample {idx}: {e}")
        
        results.append(result)
        
        # Save checkpoint
        if idx % 10 == 0 or result.get('error'):
            save_checkpoint(checkpoint, results, idx, info)
    
    # Final save
    accuracy = len([r for r in results if r.get('correct', False)]) / len(results) * 100
    info.update({'accuracy': accuracy, 'correct': sum(r.get('correct', False) for r in results)})
    
    output = f"results/{model_name.lower().replace(' ','_')}_exp{exp_type}_{input_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    
    # Calculate comprehensive metrics
    try:
        metrics = calculate_comprehensive_metrics(results)
        info['metrics'] = metrics
        print_comprehensive_results(metrics, exp_type, input_mode, len(results))
    except Exception as e:
        print(f"Metrics calculation error: {e}")
    
    with open(output, 'w') as f:
        json.dump({'info': info, 'results': results}, f, indent=2, default=str)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint): os.remove(checkpoint)
    
    print(f"\nResults: {len(results)} samples, {accuracy:.1f}% accuracy")
    print(f"Saved to: {output}")
    
    return results

def main():
    random.seed(42)
    
    parser = argparse.ArgumentParser(description="Test your model on LISTEN benchmark")
    parser.add_argument("--model", type=str, default="YourModel", help="Your model's name")
    parser.add_argument("--experiments", type=str, nargs="+", 
                       choices=["1_text", "1_audio", "1_audio_and_text", "2A", "2B", "2C", "3A", "3B", "3C", "4"],
                       help="Experiments to run (default: all)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    parser.add_argument("--list", action="store_true", help="List experiments")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nLISTEN Benchmark Experiments:")
        exps = {
            "1_text": "Text-only emotion recognition",
            "1_audio": "Audio-only emotion recognition",
            "1_audio_and_text": "Audio+Text emotion recognition",
            "2A": "Binary text classification",
            "2B": "Binary audio classification",
            "2C": "Binary audio+text classification",
            "3A": "Multi-class text (8 emotions)",
            "3B": "Multi-class audio (8 emotions)",
            "3C": "Multi-class audio+text (8 emotions)",
            "4": "Paralinguistic audio classification"
        }
        for exp, desc in exps.items():
            print(f"  {exp}: {desc}")
        return
    
    # Load model
    load_your_model()
    
    # Define experiments
    all_exps = [
        ("1_text", "text"), ("1_audio", "audio"), ("1_audio_and_text", "audio_and_text"),
        ("2A", "text"), ("2B", "audio"), ("2C", "audio_and_text"),
        ("3A", "text"), ("3B", "audio"), ("3C", "audio_and_text"),
        ("4", "audio")
    ]
    
    if args.experiments:
        exps = [(e, m) for e, m in all_exps if e in args.experiments]
    else:
        exps = all_exps
    
    print(f"\nTesting {args.model_name} on LISTEN Benchmark")
    print(f"Experiments: {', '.join([e for e, _ in exps])}")
    
    # Run experiments
    for exp_type, input_mode in exps:
        test_experiment(exp_type, input_mode, args.model_name, resume=not args.no_resume)
    
    print("\n" + "="*80)
    print("All experiments complete!")
    print("="*80)

if __name__ == "__main__":
    main()
