#!/usr/bin/env python3
"""
Comprehensive evaluation utilities for emotion classification experiments.
Includes weighted accuracy, UAR, Macro F1, Micro F1, and chance baseline calculations.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


def normalize_emotion_labels(df):
    """
    Normalize emotion labels from different datasets to standard emotion categories.
    Maps similar emotions to a common label.
    """
    # Define mapping from various dataset labels to standard emotion categories
    emotion_mapping = {
        # Anger variations
        'anger': 'anger',
        'angry': 'anger',
        
        # Happiness variations  
        'happiness': 'happiness',
        'happy': 'happiness',

        # Sadness variations
        'sad': 'sadness', 
        'sadness': 'sadness',
        
        # Surprise variations
        'surprise': 'surprise',
        'pleasant_surprise': 'surprise',
        
        # Keep other emotions as-is
        'calm': 'calm',
        'disgust': 'disgust', 
        'excitement': 'excitement',
        'fear': 'fear',
        'neutral': 'neutral',
        'frustration': 'frustration'
    }
    
    # Create a copy to avoid modifying original dataframe
    df_normalized = df.copy()
    
    # Normalize ground truth labels
    df_normalized['ground_truth'] = df_normalized['ground_truth'].map(emotion_mapping).fillna(df_normalized['ground_truth'])
    
    # Normalize prediction labels  
    df_normalized['prediction'] = df_normalized['prediction'].map(emotion_mapping).fillna(df_normalized['prediction'])
    
    return df_normalized


def prepare_dataframe_from_results(results):
    """
    Convert results list to DataFrame for evaluation.
    """
    rows = []
    for result in results:
        choices = result.get("randomized_choices") or result.get("original_choices") or []
        expected = result.get("expected_letter")
        predicted = result.get("predicted_letter")

        if not expected or not predicted:
            continue

        idx_exp = ord(expected.upper()) - ord("A")
        idx_pred = ord(predicted.upper()) - ord("A")

        if idx_exp < 0 or idx_exp >= len(choices) or idx_pred < 0 or idx_pred >= len(choices):
            continue

        gt = str(choices[idx_exp]).strip().lower()
        pred = str(choices[idx_pred]).strip().lower()

        rows.append({
            "sample_id": result.get("sample_id", ""),
            "ground_truth": gt,
            "prediction": pred,
        })
    return pd.DataFrame(rows)


def calculate_comprehensive_metrics(results):
    """
    Calculate comprehensive evaluation metrics from results.
    
    Returns:
    --------
    dict: Dictionary containing all metrics including chance baseline
    """
    df = prepare_dataframe_from_results(results)
    
    if df.empty:
        return {
            'weighted_accuracy': 0.0,
            'uar': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'per_class_accuracy': {},
            'chance_baseline': {
                'expected_accuracy': 0.0,
                'expected_macro_f1': 0.0,
                'expected_micro_f1': 0.0
            }
        }
    
    # Normalize emotion labels before calculating metrics
    df_normalized = normalize_emotion_labels(df)
    
    y_true = df_normalized["ground_truth"].tolist()
    y_pred = df_normalized["prediction"].tolist()

    # Calculate standard metrics
    wa = accuracy_score(y_true, y_pred)  # weighted accuracy = overall accuracy
    ua = balanced_accuracy_score(y_true, y_pred)  # unweighted acc (mean recall)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    per_class_acc = df_normalized.groupby("ground_truth").apply(
        lambda g: (g["ground_truth"] == g["prediction"]).mean()
    ).to_dict()
    
    # Calculate chance baseline (prediction marginal distribution baseline)
    chance_metrics = calculate_chance_baseline(df_normalized)
    
    return {
        "weighted_accuracy": wa,
        "uar": ua,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class_accuracy": per_class_acc,
        "chance_baseline": chance_metrics
    }


def calculate_chance_baseline(df):
    """
    Calculate chance baseline metrics using prediction marginal distribution.
    
    This implements the same logic as chance.py but integrated into the evaluation pipeline.
    """
    # Count predictions and ground truths
    prediction_counts = Counter(df['prediction'])
    ground_truth_counts = Counter(df['ground_truth'])
    
    if not prediction_counts or not ground_truth_counts:
        return {
            'expected_accuracy': 0.0,
            'expected_macro_f1': 0.0,
            'expected_micro_f1': 0.0
        }
    
    # Get all unique labels (union of predictions and ground truths)
    all_labels = sorted(set(prediction_counts.keys()) | set(ground_truth_counts.keys()))
    
    # Convert to numpy arrays (normalized to sum to 1)
    total_predictions = sum(prediction_counts.values())
    total_ground_truths = sum(ground_truth_counts.values())
    
    prediction_distribution = np.array([
        prediction_counts.get(label, 0) / total_predictions 
        for label in all_labels
    ])
    
    ground_truth_distribution = np.array([
        ground_truth_counts.get(label, 0) / total_ground_truths 
        for label in all_labels
    ])
    
    # Compute expected metrics
    return compute_expected_metrics(prediction_distribution, ground_truth_distribution)


def compute_expected_metrics(guess_distribution, true_distribution):
    """
    Compute expected Macro F1, Micro F1, and Accuracy for a random guessing model.
    
    This is the same function from chance.py, integrated here for convenience.
    """
    # Validate inputs
    if not np.isclose(np.sum(guess_distribution), 1.0):
        raise ValueError("guess_distribution must sum to 1")
    if not np.isclose(np.sum(true_distribution), 1.0):
        raise ValueError("true_distribution must sum to 1")
    if len(guess_distribution) != len(true_distribution):
        raise ValueError("Both distributions must have the same length")
    
    n_classes = len(true_distribution)
    
    # Expected accuracy: sum of diagonal elements of confusion matrix
    accuracy = np.sum(true_distribution * guess_distribution)
    
    # Compute per-class metrics for Macro F1
    f1_scores = []
    
    for k in range(n_classes):
        # True Positives: P(true=k) * P(predicted=k)
        tp = true_distribution[k] * guess_distribution[k]
        
        # False Positives: P(true≠k) * P(predicted=k)
        fp = (1 - true_distribution[k]) * guess_distribution[k]
        
        # False Negatives: P(true=k) * P(predicted≠k)
        fn = true_distribution[k] * (1 - guess_distribution[k])
        
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score for this class
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        f1_scores.append(f1)
    
    # Macro F1: average of per-class F1 scores
    macro_f1 = np.mean(f1_scores)
    
    # Micro F1: compute from global TP, FP, FN
    total_tp = accuracy
    total_fp = 1 - accuracy
    total_fn = 1 - accuracy
    
    # Micro precision and recall
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    # Micro F1
    if micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0
    
    return {
        'expected_accuracy': accuracy,
        'expected_macro_f1': macro_f1,
        'expected_micro_f1': micro_f1
    }


def print_comprehensive_results(metrics, experiment_type, input_mode, total_samples):
    """
    Print comprehensive evaluation results in a formatted way.
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION RESULTS")
    print(f"Experiment {experiment_type} ({input_mode.upper()})")
    print(f"{'='*80}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Weighted Accuracy (WA):   {metrics['weighted_accuracy']:.4f}")
    print(f"  UAR (Unweighted Acc):     {metrics['uar']:.4f}")
    print(f"  Macro F1:                 {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:                 {metrics['micro_f1']:.4f}")
    
    print(f"\n🎯 CHANCE BASELINE (Prediction Marginal Distribution):")
    print(f"  Expected Accuracy:        {metrics['chance_baseline']['expected_accuracy']:.4f}")
    print(f"  Expected Macro F1:        {metrics['chance_baseline']['expected_macro_f1']:.4f}")
    print(f"  Expected Micro F1:        {metrics['chance_baseline']['expected_micro_f1']:.4f}")
    
    print(f"\n📈 PERFORMANCE vs CHANCE:")
    wa_improvement = metrics['weighted_accuracy'] - metrics['chance_baseline']['expected_accuracy']
    macro_f1_improvement = metrics['macro_f1'] - metrics['chance_baseline']['expected_macro_f1']
    micro_f1_improvement = metrics['micro_f1'] - metrics['chance_baseline']['expected_micro_f1']
    
    print(f"  WA Improvement:           {wa_improvement:+.4f}")
    print(f"  Macro F1 Improvement:     {macro_f1_improvement:+.4f}")
    print(f"  Micro F1 Improvement:     {micro_f1_improvement:+.4f}")
    
    if metrics['per_class_accuracy']:
        print(f"\n📋 PER-CLASS ACCURACIES:")
        for label, acc in metrics["per_class_accuracy"].items():
            print(f"  {label:15s}: {acc:.4f}")
    
    print(f"\n📊 SAMPLE COUNT: {total_samples}")
    print(f"{'='*80}")
