import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, TensorDataset
from data_prep import build_sequences, train, val, test, feature_cols
from dlinear import train_model, evaluate


class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, window, features)
        # lstm_out shape: (batch, window, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # take only the last timestep's output
        last_out = lstm_out[:, -1, :]
        
        # pass through final linear layer
        pred = self.fc(last_out)
        return pred
    
lstm_results = {}

for window in [3, 4, 5]:
    print(f"\n--- Window {window} ---")
    
    X_train, y_train = build_sequences(train, window, feature_cols)
    X_val,   y_val   = build_sequences(val,   window, feature_cols)
    X_test,  y_test  = build_sequences(test,  window, feature_cols)
    
    model = LSTMModel(
        num_features=len(feature_cols),
        hidden_size=64,
        num_layers=1
    )
    
    model = train_model(model, X_train, y_train, X_val, y_val,
                       epochs=100, lr=0.001, batch_size=32, window=window, model_name='lstm')
    
    mse, mae = evaluate(model, X_test, y_test, window=window, model_name='lstm')
    lstm_results[f'LSTM_w{window}'] = {'MSE': mse, 'MAE': mae}

print("\nLSTM Results:")
for k, v in lstm_results.items():
    print(f"{k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}")

# Save results to file
with open('lstm_results.json', 'w') as f:
    json.dump(lstm_results, f, indent=4)
print("\nResults saved to lstm_results.json")

# --- Regularization tuning for LSTM ---
reg_params = {
    'l1': [0.0, 0.0001, 0.001],
    'l2': [0.0, 0.0001, 0.001],
    'window': [3, 4, 5],
    'lr': [0.0001, 0.0005, 0.001, 0.005],
    'epochs': [50, 100, 150, 200]
}

lstm_reg_results = {}

for window in reg_params['window']:
    for lr in reg_params['lr']:
        for epochs in reg_params['epochs']:
            for l1 in reg_params['l1']:
                for l2 in reg_params['l2']:
                    print(f"\n{'='*60}")
                    print(f"Testing LSTM: W={window}, LR={lr}, E={epochs}, L1={l1}, L2={l2}")
                    print(f"{'='*60}")
                    
                    X_train, y_train = build_sequences(train, window, feature_cols)
                    X_val, y_val = build_sequences(val, window, feature_cols)
                    X_test, y_test = build_sequences(test, window, feature_cols)
                    
                    model = LSTMModel(
                        num_features=len(feature_cols),
                        hidden_size=64,
                        num_layers=1
                    )
                    model = train_model(model, X_train, y_train, X_val, y_val, 
                                       epochs=epochs, lr=lr, batch_size=32, window=window, model_name='lstm',
                                       l1_weight=l1, l2_weight=l2)
                    
                    mse, mae = evaluate(model, X_test, y_test, window=window, model_name='lstm')
                    lstm_reg_results[f'W{window}_LR{lr}_E{epochs}_L1{l1}_L2{l2}'] = {'MSE': mse, 'MAE': mae}

print("\n" + "="*60)
print("LSTM Regularization Tuning Results:")
print("="*60)
for config, metrics in lstm_reg_results.items():
    print(f"{config}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

# save results
with open('lstm_reg_results.json', 'w') as f:
    json.dump(lstm_reg_results, f, indent=4)
print("\nLSTM regularization results saved to lstm_reg_results.json")