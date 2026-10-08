import json
import matplotlib.pyplot as plt
import numpy as np
import re

# Load results
with open('lstm_reg_results.json', 'r') as f:
    lstm_results = json.load(f)
with open('dlinear_reg_results.json', 'r') as f:
    dlinear_results = json.load(f)
with open('transformer_reg_results.json', 'r') as f:
    transformer_results = json.load(f)

def get_best_by_window(results_dict):
    """Get best MSE for each window size"""
    windows = {'W3': [], 'W4': [], 'W5': []}
    
    for config, metrics in results_dict.items():
        match = re.search(r'W[345]', config)
        if match:
            window = match.group(0)
            windows[window].append(metrics['MSE'])
    
    # Return average MSE for each window
    return {w: np.min(mse_list) for w, mse_list in windows.items() if mse_list}

# Get data for each model
dlinear_data = get_best_by_window(dlinear_results)
lstm_data = get_best_by_window(lstm_results)
transformer_data = get_best_by_window(transformer_results)

# Plot
plt.figure(figsize=(10, 6))

window_sizes = [3, 4, 5]
dlinear_mse = [dlinear_data.get(f'W{w}', np.nan) for w in window_sizes]
lstm_mse = [lstm_data.get(f'W{w}', np.nan) for w in window_sizes]
transformer_mse = [transformer_data.get(f'W{w}', np.nan) for w in window_sizes]

plt.plot(window_sizes, dlinear_mse, marker='o', label='DLinear', linewidth=2, markersize=8)
plt.plot(window_sizes, lstm_mse, marker='s', label='LSTM', linewidth=2, markersize=8)
plt.plot(window_sizes, transformer_mse, marker='^', label='Transformer', linewidth=2, markersize=8)

plt.xlabel('Window Size', fontsize=12)
plt.ylabel('Best MSE', fontsize=12)
plt.title('Model Performance vs Window Size', fontsize=14)
plt.xticks(window_sizes)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_vs_window.png', dpi=300)
plt.show()

print("Plot saved as error_vs_window.png")