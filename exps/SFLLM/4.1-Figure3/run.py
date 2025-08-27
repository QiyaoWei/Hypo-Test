import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D
from tqdm import tqdm

from dbpa.utils.setup_llm import get_responses, get_embeddings
from dbpa.data.generate_data import get_prompt
from dbpa.model.core import calculate_cosine_similarities, jensen_shannon_divergence_and_pvalue

# Main execution
baseline_prefix = "Act as a doctor. "
prefixes = ['Act as a doctor. ', 'Act as a nurse. ', 'Act as a medical practitioner. ', 'Act as a medical supervisor. ',
            'Act as a comedian. ', 'Act as a robot from the future. ', 'Act as a NeurIPS reviewer. ', 'Act as a child. ']

# Generate baseline responses
baseline_prompt = get_prompt("John", prefix=baseline_prefix)
baseline_responses = get_responses(baseline_prompt)
baseline_embeddings = get_embeddings(baseline_responses)
baseline_similarities = calculate_cosine_similarities(baseline_embeddings)

results = []

# Generate perturbed responses and calculate JSD, p-value, and JSD standard deviation
for prefix in tqdm(prefixes, desc="Processing prefixes"):
    perturbed_prompt = get_prompt("John", prefix=prefix)
    perturbed_responses = get_responses(perturbed_prompt)
    perturbed_embeddings = get_embeddings(perturbed_responses)

    # with open(f"{prefix}.json", 'w') as f:
    #     json.dump(perturbed_responses, f)
    
    perturbed_similarities = calculate_cosine_similarities(perturbed_embeddings, baseline_embeddings)
    
    # jsd, p_value, jsd_std = jensen_shannon_divergence_and_pvalue(baseline_similarities, perturbed_similarities)
    jsd, p_value = jensen_shannon_divergence_and_pvalue(baseline_embeddings, perturbed_embeddings)
    
    results.append({
        'prefix': prefix,
        'jsd': jsd,
        'p_value': p_value
    })

# Print results
print("Results:")
for result in results:
    print(f"Prefix: {result['prefix']}")
    print(f"JSD: {result['jsd']:.6f}")
    print(f"p-value: {result['p_value']:.6f}")
    print()

# Save
with open('cvd_guidelines_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to 'cvd_guidelines_results.json'")


#### PLOTTING


# Font sizes
SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 25

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Colors
SIGNIFICANT_COLOR = '#1f77b4'
INSIGNIFICANT_COLOR = '#d62728'

# Load the results from the JSON file
with open('cvd_guidelines_results.json', 'r') as f:
    results = json.load(f)

# Define the prefixes to include and their order
to_leave = ['Act as a doctor. ', 'Act as a nurse. ', 'Act as a medical practitioner. ', 'Act as a medical supervisor. ',
            'Act as a comedian. ', 'Act as a robot from the future. ', 'Act as a NeurIPS reviewer. ', 'Act as a child. ']

# Filter and sort results
filtered_results = [r for r in results if r['prefix'] in to_leave]
top_group = [r for r in filtered_results if r['prefix'] in to_leave[:4]]
bottom_group = [r for r in filtered_results if r['prefix'] in to_leave[4:]]

top_group.sort(key=lambda x: x['jsd'], reverse=True)
bottom_group.sort(key=lambda x: x['jsd'], reverse=True)

sorted_results = top_group + bottom_group

# Extract data
prefixes = [result['prefix'].replace('Act as ', '').strip().rstrip('.') for result in sorted_results]
jsds = [result['jsd'] for result in sorted_results]
p_values = [result['p_value'] for result in sorted_results]

# Create the plot
fig, ax = plt.subplots(figsize=(16, 5))

# Plot horizontal points without error bars
for i, (jsd, p_value) in enumerate(zip(jsds, p_values)):
    color = SIGNIFICANT_COLOR if p_value < 0.05 else INSIGNIFICANT_COLOR
    ax.plot(jsd, len(prefixes) - 1 - i, 'o', 
            color=color, markersize=14,
            label='Significant' if p_value < 0.05 and i == 0 else ('Not Significant' if p_value >= 0.05 and i == len(top_group) else ''))

# Customize the plot
ax.set_yticks(range(len(prefixes) - 1, -1, -1))
prefixes_title = [x.title() for x in prefixes]
ax.set_yticklabels(prefixes_title)
ax.set_xlabel(r'$\omega$', fontsize=MEDIUM_SIZE)
ax.set_title(r'$\omega$ for various input perturbations', fontsize=BIGGER_SIZE)

# Add grid lines
#ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Adjust y-axis to show all labels
plt.ylim(-1, len(prefixes))

# Add p-value annotations
for i, p_value in enumerate(p_values):
    ax.annotate(f'p = {p_value:.2f}', (max(jsds) * 1.05, len(prefixes) - 1 - i), 
                xytext=(5, 0), textcoords='offset points', 
                va='center', fontsize=SMALL_SIZE)

x_min = 0.1
x_max = 0.4
ax.set_xlim(x_min, x_max)


# Add dashed line between groups
# Add dashed line between groups
line_y = len(prefixes) - len(top_group) - 0.5
ax.axhline(y=line_y, color='black', linestyle='--', alpha=0.5)

# Add professionally-looking boxes for labels
box_props = dict(boxstyle='round,pad=0.5', facecolor='lightgrey', alpha=0.3, edgecolor='none')

# Medical professions box
ax.text(0.03, (len(prefixes) + line_y) / 2 + 1, 'Medical\nprofessions', 
        transform=ax.get_yaxis_transform(), ha='left', va='center', 
        bbox=box_props, fontsize=SMALL_SIZE, fontweight='bold')

# Random professions box
ax.text(0.03, (line_y - 1) / 2 + 1, 'Random\nprofessions', 
        transform=ax.get_yaxis_transform(), ha='left', va='center', 
        bbox=box_props, fontsize=SMALL_SIZE, fontweight='bold')


# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create custom legend with circles instead of squares
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=INSIGNIFICANT_COLOR, markersize=20, label='Not Significant'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=SIGNIFICANT_COLOR, markersize=20, label='Significant'),

]

# Add legend at the bottom, below omega, without a border
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

# Adjust layout and display the plot
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Increase bottom margin to accommodate legend
plt.savefig('0909_omega_experiment.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'cvd_guidelines_jsd_plot_filtered.png'")
