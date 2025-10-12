#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# === CONFIG ===
# Define categories with their respective experiment files
categories = {
    "Neutral Text": {
            "Text-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_experiment1_text_results_20250930_181511.json",
            "Audio-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_experiment1_audio_results_20250930_180223.json",
            "Audio+Text": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_experiment1_audio_and_text_results_20250930_184502.json",
        },
    "Emotion Matched": {
        "Text-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_experiment2A_text_results_20251002_150329.json",
        "Audio-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_2B_experiments_results_20250930_185253.json",
        "Audio+Text": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_experiment2C_audio_and_text_checkpoint.json",
    },
    "Emotion Mismatched": {
        "Text-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_3allemo_3A_experiments_results_20251001_024350.json",
        "Audio-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_3allemo_3B_experiments_results_20251001_022810.json",
        "Audio+Text": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_3allemo_3C_experiments_results_20251001_022237.json",
    },
    "Paralinguistic": {
        "Audio-only": "/users/PAS2062/delijingyic/project/VibeCheckBench/src/results/gemini2_5_pro_4_experiments_results_20250930_135802.json",
    }
}


output_dir = "figures/confusion_matrices_gemini2_5_pro_separate"

def normalize_emotion_labels(df):
    """
    Normalize emotion labels from different datasets to standard emotion categories.
    Maps similar emotions to a common label.
    """
    emotion_mapping = {
        # Anger variations
        'anger': 'anger',
        'angry': 'anger',

        # Happiness variations
        'happiness': 'happiness',
        'happy': 'happiness',
        'excitement': 'excitement',  # keep excitement separate

        # Sadness variations
        'sad': 'sadness',
        'sadness': 'sadness',

        # Surprise variations
        'surprise': 'surprise',
        'pleasant_surprise': 'surprise',

        # Keep other emotions as-is
        'calm': 'calm',
        'disgust': 'disgust',
        'fear': 'fear',
        'neutral': 'neutral',
        'frustration': 'frustration',
        'ridicule': 'ridicule',
    }

    df_norm = df.copy()
    df_norm["ground_truth"] = df_norm["ground_truth"].map(emotion_mapping).fillna(df_norm["ground_truth"])
    df_norm["prediction"] = df_norm["prediction"].map(emotion_mapping).fillna(df_norm["prediction"])
    return df_norm


def load_labels_and_preds(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "results" in data:
        results = data["results"]
    else:
        first_key = next(iter(data.keys()))
        results = data[first_key]["results"]

    # Convert letter responses to emotion names
    processed_results = []
    for r in results:
        if r.get("ground_truth") and r.get("predicted_letter") and r.get("randomized_choices"):
            # First try to use predicted_emotion if it exists and is not null
            predicted_emotion = r.get("predicted_emotion")
            
            # If predicted_emotion is null or doesn't exist, map from letter to emotion
            if not predicted_emotion:
                choices = r["randomized_choices"]
                letter_to_emotion = {}
                
                # Dynamically map letters based on available choices
                for i, choice in enumerate(choices):
                    letter = chr(ord('A') + i)  # A, B, C, D, E, etc.
                    letter_to_emotion[letter] = choice
                
                predicted_emotion = letter_to_emotion.get(r["predicted_letter"])
            
            if predicted_emotion:
                processed_results.append({
                    "ground_truth": r["ground_truth"],
                    "prediction": predicted_emotion
                })
    df = pd.DataFrame(processed_results)

    # Apply normalization
    df = normalize_emotion_labels(df)

    return df["ground_truth"].tolist(), df["prediction"].tolist()

# We'll collect labels for each individual file instead of using global labels

# Create output directory
import os
os.makedirs(output_dir, exist_ok=True)

# === CREATE COMPREHENSIVE FIGURE (excluding paralinguistic) ===
print("Creating comprehensive heatmap figure (excluding paralinguistic)...")

# Separate paralinguistic from other categories
main_categories = {k: v for k, v in categories.items() if k != "Paralinguistic"}
paralinguistic_data = categories.get("Paralinguistic", {})

# Collect all labels across main categories for consistent labeling
all_labels = set()
for category_data in main_categories.values():
    for path in category_data.values():
        y_true, y_pred = load_labels_and_preds(path)
        all_labels.update(y_true)
        all_labels.update(y_pred)
all_labels = sorted(list(all_labels))
print(f"Total unique labels across main categories: {len(all_labels)}")
print(f"Labels: {all_labels}")

# Calculate grid dimensions for main categories
n_categories = len(main_categories)
max_modalities = max(len(category_data) for category_data in main_categories.values())
print(f"Grid: {n_categories} rows × {max_modalities} columns")

# Create one large figure for main categories
fig, axes = plt.subplots(n_categories, max_modalities, figsize=(6 * max_modalities, 5 * n_categories))

# Handle single row case
if n_categories == 1:
    axes = axes.reshape(1, -1)

# Create heatmaps for each main category
for row_idx, (category_name, category_data) in enumerate(main_categories.items()):
    print(f"Processing {category_name}...")
    
    for col_idx in range(max_modalities):
        ax = axes[row_idx, col_idx]
        
        if col_idx < len(category_data):
            # Get modality and path
            modality = list(category_data.keys())[col_idx]
            path = list(category_data.values())[col_idx]
            
            # Load data
            y_true, y_pred = load_labels_and_preds(path)
            
            # Create confusion matrix with all labels
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)

            # Row-normalize (distribution across predicted classes for each true class)
            with np.errstate(all='ignore'):  # suppress warnings on division by zero
                cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)  # replace NaNs with 0

            # Calculate dynamic annotation font size
            n_labels = len(all_labels)
            annot_size = max(3, min(6, int(60 / max(1, n_labels))))
            
            sns.heatmap(
                cm_normalized,
                annot=False,
                cmap="Blues",
                vmin=0, vmax=1,
                xticklabels=all_labels if row_idx == n_categories-1 else [],  # show x-labels on bottom row
                yticklabels=all_labels if col_idx == 0 else [],  # show y-labels on left column
                cbar=(col_idx == max_modalities-1),  # only rightmost column shows colorbar
                ax=ax
            )
            ax.set_title(f"{category_name} – {modality}", fontsize=15)
            if col_idx == 0:
                ax.set_ylabel("Ground Truth", fontsize=16)
            if row_idx == n_categories-1:
                ax.set_xlabel("Predicted", fontsize=16)
            # Slightly enlarge tick labels
            ax.tick_params(axis='both', which='major', labelsize=14)
        else:
            # Empty subplot
            ax.set_visible(False)

plt.tight_layout()

# Save the comprehensive figure
output_file = f"{output_dir}/confusion_matrices_gemini2_5_pro_main_categories.pdf"
plt.savefig(output_file, format='pdf', dpi=600, bbox_inches='tight')
print(f"✅ Saved main categories confusion matrices to {output_file}")
plt.close()

# === CREATE SEPARATE PARALINGUISTIC FIGURE ===
if paralinguistic_data:
    print("Creating separate paralinguistic heatmap...")
    
    # Get paralinguistic labels
    paralinguistic_labels = set()
    for path in paralinguistic_data.values():
        y_true, y_pred = load_labels_and_preds(path)
        paralinguistic_labels.update(y_true)
        paralinguistic_labels.update(y_pred)
    paralinguistic_labels = sorted(list(paralinguistic_labels))
    print(f"Paralinguistic labels: {paralinguistic_labels}")
    
    # Create separate figure for paralinguistic
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Get paralinguistic data
    modality = list(paralinguistic_data.keys())[0]
    path = list(paralinguistic_data.values())[0]
    y_true, y_pred = load_labels_and_preds(path)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=paralinguistic_labels)
    
    # Row-normalize
    with np.errstate(all='ignore'):
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # Calculate annotation font size
    n_labels = len(paralinguistic_labels)
    annot_size = max(6, min(10, int(100 / max(1, n_labels))))
    
    sns.heatmap(
        cm_normalized,
        annot=False,
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=paralinguistic_labels,
        yticklabels=paralinguistic_labels,
        cbar=True,
        ax=ax
    )
    ax.set_title(f"Gemini-2.5-Pro – Paralinguistic – {modality}", fontsize=16)
    ax.set_ylabel("Ground Truth", fontsize=16)
    ax.set_xlabel("Predicted", fontsize=16)
    # Slightly enlarge tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    
    # Save paralinguistic figure
    paralinguistic_output = f"{output_dir}/confusion_matrix_paralinguistic_separate.pdf"
    plt.savefig(paralinguistic_output, format='pdf', dpi=600, bbox_inches='tight')
    print(f"✅ Saved paralinguistic confusion matrix to {paralinguistic_output}")
    plt.close()

print(f"✅ All confusion matrices saved to {output_dir}/")
