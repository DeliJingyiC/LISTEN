import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data from the table - extracting key performance metrics
# Using Overall Average column as the main metric for each model
models = [
    'Qwen3-instruct',
    'Qwen2.5-Omni-7B', 
    'Qwen3-Omni-30B',
    'Baichuan-Omni-1.5',
    'Gemini2.5-Flash',
    'Gemini2.5-Pro'
]

# Categories based on the experimental conditions
categories = [
    'Neutral-Text\nAverage',
    'Emotion-Matched\nAverage', 
    'Emotion-Mismatched\nAverage',
    'Paralinguistic\nAudio',
    'Overall\nAverage'
]

# Performance data (Overall Average column from table)
performance_data = {
    'Qwen3-instruct': [66.6, 33.5, 38.0, 0, 34.4],  # No audio data, estimated overall
    'Qwen2.5-Omni-7B': [26.9, 37.6, 38.8, 22.7, 32.8],
    'Qwen3-Omni-30B': [27.3, 42.8, 38.3, 21.0, 33.9],
    'Baichuan-Omni-1.5': [15.9, 36.0, 36.0, 22.7, 28.3],
    'Gemini2.5-Flash': [25.1, 34.8, 36.9, 18.0, 30.2],
    'Gemini2.5-Pro': [38.3, 38.9, 39.8, 15.7, 35.7]
}

# Create the radar chart
def create_radar_chart():
    # Number of variables
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for different models
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Plot each model
    for i, (model, values) in enumerate(performance_data.items()):
        # Complete the circle
        values_plot = values + values[:1]
        angles_plot = angles + angles[:1]
        
        # Plot the line
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, 
                label=model, color=colors[i], markersize=6)
        
        # Fill the area
        ax.fill(angles_plot, values_plot, alpha=0.1, color=colors[i])
    
    # Customize the chart
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 50)  # Adjust based on your data range
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add title and legend
    plt.title('Model Performance Comparison Across Different Tasks\n', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_performance_radar_basic.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Alternative: Create a more detailed radar chart with separate metrics
def create_detailed_radar_chart():
    # More detailed categories based on individual modalities
    detailed_categories = [
        'Neutral-Text\nText',
        'Neutral-Text\nAudio', 
        'Emotion-Matched\nText',
        'Emotion-Matched\nAudio',
        'Emotion-Mismatched\nText',
        'Emotion-Mismatched\nAudio',
        'Paralinguistic\nAudio'
    ]
    
    # More detailed performance data
    detailed_performance = {
        'Qwen2.5-Omni-7B': [85.4, 34.0, 36.4, 36.6, 34.0, 38.5, 22.7],
        'Qwen3-Omni-30B': [85.4, 29.3, 38.7, 42.4, 34.6, 37.4, 21.0],
        'Baichuan-Omni-1.5': [81.2, 16.5, 31.0, 36.0, 31.0, 36.0, 22.7],
        'Gemini2.5-Flash': [82.5, 25.6, 36.6, 30.7, 33.1, 35.8, 18.0],
        'Gemini2.5-Pro': [96.6, 34.9, 38.8, 37.6, 31.8, 36.9, 15.7]
    }
    
    N = len(detailed_categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (model, values) in enumerate(detailed_performance.items()):
        values_plot = values + values[:1]
        angles_plot = angles + angles[:1]
        
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, 
                label=model, color=colors[i], markersize=7)
        ax.fill(angles_plot, values_plot, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles)
    ax.set_xticklabels(detailed_categories, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.title('Detailed Model Performance Across Modalities\n', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=11)
    
    plt.tight_layout()
    plt.savefig('model_performance_radar_detailed.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Create a comparison table visualization
def create_performance_table():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Model', 'Neutral-Text Avg', 'Emotion-Matched Avg', 
               'Emotion-Mismatched Avg', 'Paralinguistic Audio', 'Overall Avg']
    
    for model in models:
        if model in performance_data:
            row = [model] + [f"{val:.1f}%" if val > 0 else "--" for val in performance_data[model]]
            table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best performance in each column
    for col in range(1, len(headers)):
        values = [float(row[col].replace('%', '').replace('--', '0')) for row in table_data]
        max_val = max(values)
        max_idx = values.index(max_val) + 1  # +1 for header row
        
        table[(max_idx, col)].set_facecolor('#FFD700')  # Gold for best
        table[(max_idx, col)].set_text_props(weight='bold')
    
    plt.title('Model Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('model_performance_table.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Creating radar chart visualizations and saving as PDF...")
    print("\n1. Basic Performance Comparison:")
    create_radar_chart()
    print("   ✅ Saved as: model_performance_radar_basic.pdf")
    
    print("\n2. Detailed Modality Performance:")
    create_detailed_radar_chart()
    print("   ✅ Saved as: model_performance_radar_detailed.pdf")
    
    print("\n3. Performance Summary Table:")
    create_performance_table()
    print("   ✅ Saved as: model_performance_table.pdf")
    
    print("\n🎉 All figures have been saved as high-quality PDF files!")
