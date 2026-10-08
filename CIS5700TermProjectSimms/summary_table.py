import json
import pandas as pd
import re

# load all results
with open('lstm_reg_results.json', 'r') as f:
    lstm_results = json.load(f)

with open('dlinear_reg_results.json', 'r') as f:
    dlinear_results = json.load(f)

with open('transformer_reg_results.json', 'r') as f:
    transformer_results = json.load(f)

def get_top_results(results_dict, model_name, n=10):
    rows = []
    for config, metrics in results_dict.items():
        row = {'Model': model_name, 'Config': config}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('MSE').head(n)
    return df

# get top 10 for each model
top_lstm        = get_top_results(lstm_results, 'LSTM', n=10)
top_dlinear     = get_top_results(dlinear_results, 'DLinear', n=10)
top_transformer = get_top_results(transformer_results, 'Transformer', n=10)

# combine into one table
combined = pd.concat([top_dlinear, top_lstm, top_transformer])
combined = combined.round(4)

print("\nTop 10 Configurations Per Model (sorted by MSE):")
print(combined.to_string(index=False))

# also print the single best per model for quick comparison
print("\n--- Best Configuration Per Model ---")
for df, name in [(top_dlinear, 'DLinear'), 
                  (top_lstm, 'LSTM'), 
                  (top_transformer, 'Transformer')]:
    best = df.iloc[0]
    print(f"{name}: Config={best['Config']} "
          f"MSE={best['MSE']:.4f} MAE={best['MAE']:.4f}")
    

def get_top_by_window(results_dict, model_name, n=5):
    rows = []
    for config, metrics in results_dict.items():
        match = re.search(r'W[345]', config)
        window = match.group(0) if match else 'Unknown'
        row = {'Model': model_name, 'Window': window, 'Config': config}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    top = (df.groupby('Window')
             .apply(lambda x: x.nsmallest(n, 'MSE'))
             .reset_index(drop=True))
    return top

top_lstm_w        = get_top_by_window(lstm_results, 'LSTM')
top_dlinear_w     = get_top_by_window(dlinear_results, 'DLinear')
top_transformer_w = get_top_by_window(transformer_results, 'Transformer')

combined_w = pd.concat([top_dlinear_w, top_lstm_w, top_transformer_w])
combined_w = combined_w.sort_values(['Model', 'Window', 'MSE']).round(4).reset_index(drop=True)

print("\nTop 5 Per Window Per Model:")
print(combined_w.to_string(index=False))

# --- Summary table: best per model per window ---
summary_rows = []

for results, name in [(dlinear_results, 'DLinear'),
                       (lstm_results, 'LSTM'),
                       (transformer_results, 'Transformer')]:
    for window in ['W3', 'W4', 'W5']:
        window_results = {k: v for k, v in results.items()
                         if re.search(r'W[345]', k) and
                         re.search(r'W[345]', k).group(0) == window}
        if window_results:
            best_config = min(window_results,
                            key=lambda x: window_results[x]['MSE'])
            summary_rows.append({
                'Model': name,
                'Window': window,
                'MSE': round(window_results[best_config]['MSE'], 4),
                'MAE': round(window_results[best_config]['MAE'], 4)
            })

summary_df = pd.DataFrame(summary_rows)
print("\nSummary Table:")
print(summary_df.to_string(index=False))

