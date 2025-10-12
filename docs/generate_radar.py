#!/usr/bin/env python3
"""
Generate radar chart for LISTEN benchmark results
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Model data from the leaderboard (Overall Average)
models = {
    'Gemini 2.5 Pro': 35.7,
    'Qwen3-Omni-30B': 33.9,
    'Qwen2.5-Omni-7B': 32.8,
    'Gemini 2.5 Flash': 30.2,
    'Baichuan-Omni-1.5': 28.3
}

# Experiment data (Audio and Text+Audio averages for each condition)
experiment_data = {
    'Gemini 2.5 Pro': {
        'Neutral-Text': (34.9 + 41.7) / 2,  # Audio + Both
        'Emotion-Matched': (37.6 + 40.2) / 2,
        'Emotion-Mismatched': (36.9 + 42.6) / 2,
        'Paralinguistic': 15.7
    },
    'Qwen3-Omni-30B': {
        'Neutral-Text': (29.3 + 25.3) / 2,
        'Emotion-Matched': (42.4 + 43.1) / 2,
        'Emotion-Mismatched': (37.4 + 39.1) / 2,
        'Paralinguistic': 21.0
    },
    'Qwen2.5-Omni-7B': {
        'Neutral-Text': (34.0 + 19.8) / 2,
        'Emotion-Matched': (36.6 + 38.6) / 2,
        'Emotion-Mismatched': (38.5 + 39.1) / 2,
        'Paralinguistic': 22.7
    },
    'Gemini 2.5 Flash': {
        'Neutral-Text': (25.6 + 24.6) / 2,
        'Emotion-Matched': (30.7 + 38.9) / 2,
        'Emotion-Mismatched': (35.8 + 38.0) / 2,
        'Paralinguistic': 18.0
    },
    'Baichuan-Omni-1.5': {
        'Neutral-Text': (16.5 + 15.2) / 2,
        'Emotion-Matched': (36.0 + 36.0) / 2,
        'Emotion-Mismatched': (36.0 + 36.0) / 2,
        'Paralinguistic': 22.7
    }
}

# Categories
categories = ['Neutral-Text', 'Emotion-Matched', 'Emotion-Mismatched', 'Paralinguistic']
N = len(categories)

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='polar')

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Colors for each model (matching website theme)
colors = {
    'Gemini 2.5 Pro': '#2563eb',      # Blue
    'Qwen3-Omni-30B': '#7c3aed',      # Purple
    'Qwen2.5-Omni-7B': '#06b6d4',     # Cyan
    'Gemini 2.5 Flash': '#10b981',    # Green
    'Baichuan-Omni-1.5': '#f59e0b'    # Orange
}

# Plot data for each model
for model_name, color in colors.items():
    values = [experiment_data[model_name][cat] for cat in categories]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)

# Fix axis to go in the right order and start at 12 o'clock
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold')

# Set y-axis limits and labels
ax.set_ylim(0, 50)
ax.set_yticks([10, 20, 30, 40, 50])
ax.set_yticklabels(['10%', '20%', '30%', '40%', '50%'], size=9, color='gray')

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=True, shadow=True)

# Add title
plt.title('Model Performance Across Experimental Conditions', 
          size=14, weight='bold', pad=20)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('/users/PAS2062/delijingyic/project/LISTEN/docs/performance_radar.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
print("✓ Radar chart saved to docs/performance_radar.png")

plt.close()

