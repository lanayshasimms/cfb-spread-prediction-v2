import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, TensorDataset
from data_prep import build_sequences, train, val, test, feature_cols
from dlinear import train_model, evaluate


class VanillaTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, 
                 num_layers=1, dropout=0.1):
        super().__init__()
        
        # project input features into d_model dimensions
        self.input_projection = nn.Linear(num_features, d_model)
        
        # positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # final linear layer to predict spread
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch, window, features)
        
        # project to d_model
        x = self.input_projection(x)
        
        # add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # pass through transformer encoder
        x = self.transformer(x)
        
        # take last timestep and predict
        out = self.fc(x[:, -1, :])
        return out


transformer_results = {}

for window in [3, 4, 5]:
    print(f"\n--- Window {window} ---")
    
    X_train, y_train = build_sequences(train, window, feature_cols)
    X_val,   y_val   = build_sequences(val,   window, feature_cols)
    X_test,  y_test  = build_sequences(test,  window, feature_cols)
    
    model = VanillaTransformer(
        num_features=len(feature_cols),
        d_model=64,
        nhead=4,
        num_layers=1,
        dropout=0.1
    )
    
    model = train_model(model, X_train, y_train, X_val, y_val,
                       epochs=100, lr=0.001, batch_size=32, 
                       window=window, model_name='transformer')
    
    mse, mae = evaluate(model, X_test, y_test, 
                        window=window, model_name='transformer')
    transformer_results[f'Transformer_w{window}'] = {'MSE': mse, 'MAE': mae}

print("\nTransformer Results:")
for k, v in transformer_results.items():
    print(f"{k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}")

with open('transformer_results.json', 'w') as f:
    json.dump(transformer_results, f, indent=4)
print("\nResults saved to transformer_results.json")


# --- Regularization tuning for Transformer ---
reg_params = {
    'l1': [0.0, 0.0001, 0.001],
    'l2': [0.0, 0.0001, 0.001],
    'window': [3, 4, 5],
    'lr': [0.0001, 0.0005, 0.001, 0.005],
    'epochs': [50, 100, 150, 200]
}

transformer_reg_results = {}

for window in reg_params['window']:
    for lr in reg_params['lr']:
        for epochs in reg_params['epochs']:
            for l1 in reg_params['l1']:
                for l2 in reg_params['l2']:
                    print(f"\n{'='*60}")
                    print(f"Testing Transformer: W={window}, LR={lr}, E={epochs}, L1={l1}, L2={l2}")
                    print(f"{'='*60}")
                    
                    X_train, y_train = build_sequences(train, window, feature_cols)
                    X_val, y_val = build_sequences(val, window, feature_cols)
                    X_test, y_test = build_sequences(test, window, feature_cols)
                    
                    model = VanillaTransformer(
                        num_features=len(feature_cols),
                        d_model=64,
                        nhead=4,
                        num_layers=1,
                        dropout=0.1
                    )
                    model = train_model(model, X_train, y_train, X_val, y_val, 
                                       epochs=epochs, lr=lr, batch_size=32, window=window, model_name='transformer',
                                       l1_weight=l1, l2_weight=l2)
                    
                    mse, mae = evaluate(model, X_test, y_test, window=window, model_name='transformer')
                    transformer_reg_results[f'W{window}_LR{lr}_E{epochs}_L1{l1}_L2{l2}'] = {'MSE': mse, 'MAE': mae}

# save regularization results
with open('transformer_reg_results.json', 'w') as f:
    json.dump(transformer_reg_results, f, indent=4)
print("\nTransformer regularization results saved to transformer_reg_results.json")