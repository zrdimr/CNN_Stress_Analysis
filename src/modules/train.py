import os
import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, config, run_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    best_val_loss = float('inf')
    
    run_save_dir = os.path.join(config['training']['save_dir'], run_name)
    os.makedirs(run_save_dir, exist_ok=True)
    save_path = os.path.join(run_save_dir, 'best_model.pth')

    print(f"Starting training on {device} for {run_name}...")
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += y_batch.size(0)
            correct_train += (predicted == y_batch).sum().item()
            
        train_acc = 100 * correct_train / total_train
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += y_batch.size(0)
                correct_val += (predicted == y_batch).sum().item()
                
        val_acc = 100 * correct_val / total_val
        val_loss_avg = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), save_path)
            
    print(f"=> Training complete. Best model saved to {save_path}")
    return save_path
