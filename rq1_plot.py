import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ['CodeBERT', 'GraphCodeBERT', 'UniXcoder', 'CodeT5']
metrics = ['Accuracy', 'Recall', 'Precision', 'F1']

# Devign and Reveal dataset results
devign_data = {
    'CodeBERT': [0.6332, 0.3554, 0.698, 0.471],
    'DFEPT+CodeBERT': [0.6475, 0.5004, 0.6162, 0.5523],
    'GraphCodeBERT': [0.6384, 0.3665, 0.7044, 0.4822],
    'DFEPT+GraphCodeBERT': [0.6296, 0.5227, 0.6137, 0.5645],
    'UniXcoder': [0.6369, 0.4765, 0.6409, 0.5466],
    'DFEPT+UniXcoder': [0.6497, 0.5108, 0.6514, 0.5726],
    'CodeT5': [0.6336, 0.5952, 0.6024, 0.5988],
    'DFEPT+CodeT5': [0.6453, 0.6096, 0.615, 0.6122]
}

reveal_data = {
    'CodeBERT': [0.9085, 0.2609, 0.6122, 0.3659],
    'DFEPT+CodeBERT': [0.9125, 0.2652, 0.6703, 0.3801],
    'GraphCodeBERT': [0.9156, 0.287, 0.7021, 0.4074],
    'DFEPT+GraphCodeBERT': [0.9265, 0.2952, 0.7654, 0.4261],
    'UniXcoder': [0.9094, 0.3087, 0.6017, 0.408],
    'DFEPT+UniXcoder': [0.9182, 0.3261, 0.7075, 0.4464],
    'CodeT5': [0.9265, 0.3524, 0.7048, 0.4698],
    'DFEPT+CodeT5': [0.9292, 0.3524, 0.7475, 0.479]
}

# Custom colors for the bars
colors = ['#B02226', '#F0A12C', '#5EA0C7', '#009E73']

# Adjusted plot function with custom colors and black borders
def plot_comparison_with_borders(ax, data_original, data_dfept, title, opt=1):
    n_groups = 2  # Number of groups (original model and model with DFEPT)
    index = np.arange(n_groups) * 1.2  # Group positions with increased spacing
    bar_width = 0.2  # Width of the bars

    # Find min and max values to set y-axis limits
    all_values = data_original + data_dfept
    lower_limit = min(all_values) - 0.1 * (max(all_values) - min(all_values))
    upper_limit1 = max(all_values) + 0.3 * (max(all_values) - min(all_values))
    upper_limit2 = max(all_values) + 0.05 * (max(all_values) - min(all_values))

    # Plotting each metric with custom colors and black borders
    for i, metric in enumerate(metrics):
        original_metrics = [data_original[i], data_dfept[i]]  # Metrics for the original and DFEPT models
        ax.bar(index + i * bar_width, original_metrics, bar_width, label=metric, color=colors[i], edgecolor='black')

    ax.set_title(title, fontsize=40)
    if opt ==1:
        ax.set_ylim([lower_limit, upper_limit1])
    else:
        ax.set_ylim([lower_limit, upper_limit2])
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels(['Only Fine-tuned', 'DFEPT'], fontsize=32)
    ax.legend(prop={'size': 25})
    ax.tick_params(axis='y', labelsize=32)

# Preparing the figure and axes
fig, axs = plt.subplots(2, 4, figsize=(45, 20), constrained_layout=True)

# Plotting each model comparison on Devign and Reveal datasets with custom colors and black borders
for i, model in enumerate(models):
    plot_comparison_with_borders(axs[0, i], devign_data[model], devign_data['DFEPT+' + model], f'{model} on Devign')
    plot_comparison_with_borders(axs[1, i], reveal_data[model], reveal_data['DFEPT+' + model], f'{model} on Reveal',opt=2)

# Saving the figure as a PDF file with 300 DPI
fig.savefig("Fig4.pdf", dpi=300)
plt.show()