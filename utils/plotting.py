import json
import matplotlib.pyplot as plt
import numpy as np
import os # Import the os module

# Define the output directory
output_dir = "output/"
results_file = os.path.join(output_dir, "test_results.json")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# --- Load the JSON data from the file ---
try:
    with open(results_file, 'r') as f:
        data = json.load(f)
    metrics = data['metrics']
    print(f"Successfully loaded results from {results_file}")
except FileNotFoundError:
    print(f"Error: Results file not found at {results_file}")
    print("Please ensure 'test_results.json' is in the 'output' directory.")
    exit() # Exit if the file doesn't exist
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {results_file}")
    print("Please check the file format.")
    exit()
except KeyError:
    print(f"Error: Expected 'metrics' key not found in {results_file}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading data: {e}")
    exit()


# --- Plot 1: Level Accuracy and Path Similarity ---
try:
    level_accuracy = metrics['level_accuracy']
    path_similarity = metrics['path_similarity']

    levels = list(level_accuracy.keys())
    accuracy_values = list(level_accuracy.values())

    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
    plt.figure(figsize=(10, 6))
    bars = plt.bar(levels, accuracy_values, color='skyblue', label='Level Accuracy')

    # Add a horizontal line for path similarity
    plt.axhline(path_similarity, color='red', linestyle='--', linewidth=2,
                label=f'Path Similarity ({path_similarity:.3f})')

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f'{yval:.3f}',
                 va='bottom', ha='center', fontsize=9) # Adjust position and size

    plt.ylim(0, 1.05) # Give some space above 1.0
    plt.xlabel('Hierarchy Level', fontsize=12)
    plt.ylabel('Accuracy / Similarity Score', fontsize=12)
    plt.title('Hierarchical Classification Performance: Level Accuracy vs. Path Similarity', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot instead of showing it
    plot1_filename = os.path.join(output_dir, "level_accuracy_vs_path_similarity.png")
    plt.savefig(plot1_filename)
    print(f"Plot 1 saved to {plot1_filename}")
    plt.close() # Close the figure to free memory

except KeyError as e:
    print(f"Error plotting Level Accuracy: Missing key {e} in metrics data.")
except Exception as e:
    print(f"An error occurred during Plot 1 generation: {e}")


# --- Plot 2: Aggregated Metrics per Level ---
try:
    level_metrics = metrics['level_metrics']

    # Ensure 'levels' is defined from Plot 1 (if Plot 1 failed, this might error)
    if 'levels' not in locals():
         print("Error: 'levels' variable not defined (likely due to Plot 1 failure). Cannot create Plot 2.")
         exit()

    # Extract P, R, F1 for each level (handle potential missing levels if needed)
    precisions = [level_metrics.get(level, {}).get('precision', 0) for level in levels]
    recalls = [level_metrics.get(level, {}).get('recall', 0) for level in levels]
    f1_scores = [level_metrics.get(level, {}).get('f1', 0) for level in levels]

    x = np.arange(len(levels))  # the label locations
    width = 0.25  # the width of the bars

    plt.figure(figsize=(12, 7))
    rects1 = plt.bar(x - width, precisions, width, label='Precision', color='cornflowerblue')
    rects2 = plt.bar(x, recalls, width, label='Recall', color='lightcoral')
    rects3 = plt.bar(x + width, f1_scores, width, label='F1 Score', color='lightgreen')

    # Add some text for labels, title and axes ticks
    plt.ylabel('Scores', fontsize=12)
    plt.xlabel('Hierarchy Level', fontsize=12)
    plt.title('Average Precision, Recall, and F1 Score per Hierarchy Level', fontsize=14)
    plt.xticks(x, levels)
    plt.ylim(0, 1.1) # Give space above 1.0 for labels
    plt.legend(fontsize=10, loc='upper right')

    # Function to add labels to grouped bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            # Only add label if height > 0 to avoid cluttering zero bars
            if height > 0.01: # Use a small threshold
                plt.annotate(f'{height:.2f}', # Format to 2 decimal places
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, rotation=90) # Rotate labels

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot instead of showing it
    plot2_filename = os.path.join(output_dir, "average_metrics_per_level.png")
    plt.savefig(plot2_filename)
    print(f"Plot 2 saved to {plot2_filename}")
    plt.close() # Close the figure to free memory

except KeyError as e:
    print(f"Error plotting Level Metrics: Missing key {e} in metrics data.")
except Exception as e:
    print(f"An error occurred during Plot 2 generation: {e}")

print(f"\nPlots saved in '{output_dir}'. Detailed node performance data is available in the 'node_performance' key but not plotted individually due to the large number of nodes.")