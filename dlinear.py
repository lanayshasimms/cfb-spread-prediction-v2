import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_prep import build_sequences, train, val, test, feature_cols
import json


# --- moving average for decomposition ---
class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, 
                                stride=1, 
                                padding=0)
    
    def forward(self, x):
        # x shape: (batch, window, features)
        # pad both ends to keep sequence length
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        x_padded = torch.cat([
            x[:, :1, :].repeat(1, pad_left, 1),
            x,
            x[:, -1:, :].repeat(1, pad_right, 1)
        ], dim=1)
        # pool over time dimension
        # AvgPool1d expects (batch, features, time)
        x_padded = x_padded.permute(0, 2, 1)
        trend = self.avg(x_padded)
        trend = trend.permute(0, 2, 1)
        return trend

# --- DLinear model ---
class DLinear(nn.Module):
    def __init__(self, window_size, num_features, kernel_size=3):
        super().__init__()
        self.decomp = MovingAverage(kernel_size)
        
        # one linear layer per component
        # input is flattened: window_size * num_features
        self.trend_layer    = nn.Linear(window_size * num_features, 1)
        self.remainder_layer = nn.Linear(window_size * num_features, 1)
    
    def forward(self, x):
        # x shape: (batch, window, features)
        trend = self.decomp(x)
        remainder = x - trend
        
        # flatten for linear layer
        trend_flat     = trend.reshape(trend.size(0), -1)
        remainder_flat = remainder.reshape(remainder.size(0), -1)
        
        trend_pred     = self.trend_layer(trend_flat)
        remainder_pred = self.remainder_layer(remainder_flat)
        
        return trend_pred + remainder_pred

# --- Training function ---
"""def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, lr=0.001, batch_size=32, window=3, model_name='dlinear'):
    
    # convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    model_path = f'best_{model_name}_w{window}.pt'
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} "
                  f"Train Loss: {train_loss/len(loader):.4f} "
                  f"Val Loss: {val_loss:.4f}")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
    
    return model """


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=100, lr=0.001, batch_size=32, window=3, model_name='dlinear',
                l1_weight=0.0, l2_weight=0.0):
    
    # convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.FloatTensor(y_val).unsqueeze(1)
    
    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    model_path = f'best_{model_name}_w{window}.pt'
    
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            
            # add l1 regularization
            if l1_weight > 0:
                l1_reg = torch.tensor(0.)
                for param in model.parameters():
                    l1_reg += torch.sum(torch.abs(param))
                loss += l1_weight * l1_reg
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} "
                  f"Train Loss: {train_loss/len(loader):.4f} "
                  f"Val Loss: {val_loss:.4f}")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
    
    return model

# --- Evaluation function ---
def evaluate(model, X_test, y_test, window=3, model_name='dlinear'):
    model_path = f'best_{model_name}_w{window}.pt'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    with torch.no_grad():
        pred = model(X_test_t)
        mse  = nn.MSELoss()(pred, y_test_t).item()
        mae  = nn.L1Loss()(pred, y_test_t).item()
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    return mse, mae 



if __name__ == '__main__':
    results = {}

    for window in [3, 4, 5]:
        print(f"\n--- Window {window} ---")
        
        X_train, y_train = build_sequences(train, window, feature_cols)
        X_val,   y_val   = build_sequences(val,   window, feature_cols)
        X_test,  y_test  = build_sequences(test,  window, feature_cols)
        
        model = DLinear(window_size=window, num_features=len(feature_cols), kernel_size=3)
    
    
        # pass window size to train_model so it saves unique filenames
        model = train_model(model, X_train, y_train, X_val, y_val, 
                        epochs=100, lr=0.001, batch_size=32, window=window)
        

        mse, mae = evaluate(model, X_test, y_test, window=window)    
        results[f'DLinear_w{window}'] = {'MSE': mse, 'MAE': mae}

    print("\nFinal Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # Save results to a file
    with open('dlinear_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to dlinear_results.json")



    # --- Regularization tuning ---
    reg_params = {
        'l1': [0.0, 0.0001, 0.001],
        'l2': [0.0, 0.0001, 0.001],
        'window': [3, 4, 5],
        'lr': [0.0001, 0.0005, 0.001, 0.005],
        'epochs': [50, 100, 150, 200]
    }

    reg_results = {}

    for window in reg_params['window']:
        for lr in reg_params['lr']:
            for epochs in reg_params['epochs']:
                for l1 in reg_params['l1']:
                    for l2 in reg_params['l2']:
                        print(f"\n{'='*60}")
                        print(f"Testing: W={window}, LR={lr}, E={epochs}, L1={l1}, L2={l2}")
                        print(f"{'='*60}")
                        
                        X_train, y_train = build_sequences(train, window, feature_cols)
                        X_val, y_val = build_sequences(val, window, feature_cols)
                        X_test, y_test = build_sequences(test, window, feature_cols)
                        
                        model = DLinear(window_size=window, num_features=len(feature_cols), kernel_size=3)
                        model = train_model(model, X_train, y_train, X_val, y_val, 
                                        epochs=epochs, lr=lr, batch_size=32, window=window,
                                        l1_weight=l1, l2_weight=l2)
                        
                        mse, mae = evaluate(model, X_test, y_test, window=window)
                        reg_results[f'W{window}_LR{lr}_E{epochs}_L1{l1}_L2{l2}'] = {'MSE': mse, 'MAE': mae}

    print("\n" + "="*60)
    print("DLinear Regularization Tuning Results:")
    print("="*60)
    for config, metrics in reg_results.items():
        print(f"{config}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

    # save results
    with open('dlinear_reg_results.json', 'w') as f:
        json.dump(reg_results, f, indent=4)
    print("\nRegularization results saved to dlinear_reg_results.json")