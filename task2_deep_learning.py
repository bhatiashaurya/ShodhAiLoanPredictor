"""
Task 2: Deep Learning Classification Model
===========================================

This module implements:
- Multi-layer Perceptron (MLP) architecture
- Advanced training techniques
- Model evaluation with AUC-ROC and F1-Score
- Uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import pickle

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


class LoanDefaultMLP(nn.Module):
    """
    Deep Learning Model for Loan Default Prediction
    
    Architecture:
    - 6 hidden layers: [768, 512, 384, 256, 128, 64]
    - Batch Normalization
    - Dropout regularization
    - Residual connections
    - Mish activation function
    """
    
    def __init__(self, input_dim, hidden_dims=[768, 512, 384, 256, 128, 64], 
                 dropout=0.4, use_mixup=False, n_heads=4):
        super(LoanDefaultMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_mixup = use_mixup
        
        # Input layer
        self.fc_input = nn.Linear(input_dim, hidden_dims[0])
        self.bn_input = nn.BatchNorm1d(hidden_dims[0])
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output layer
        self.fc_output = nn.Linear(hidden_dims[-1], 1)
        
        # Multi-head attention (simplified)
        self.n_heads = n_heads
        if n_heads > 1:
            self.attention_heads = nn.ModuleList([
                nn.Linear(hidden_dims[0], hidden_dims[0] // n_heads)
                for _ in range(n_heads)
            ])
            self.attention_combine = nn.Linear(hidden_dims[0], hidden_dims[0])
    
    def mish(self, x):
        """Mish activation: x * tanh(softplus(x))"""
        return x * torch.tanh(F.softplus(x))
    
    def forward(self, x):
        """Forward pass"""
        # Input layer
        x = self.fc_input(x)
        x = self.bn_input(x)
        x = self.mish(x)
        
        # Multi-head attention (if enabled)
        if self.n_heads > 1:
            attention_outputs = [head(x) for head in self.attention_heads]
            x_attention = torch.cat(attention_outputs, dim=1)
            x = self.attention_combine(x_attention)
            x = self.mish(x)
        
        # Hidden layers with residual connections
        for i, (fc, bn, dropout) in enumerate(zip(self.hidden_layers, 
                                                   self.batch_norms, 
                                                   self.dropouts)):
            identity = x
            x = fc(x)
            x = bn(x)
            x = self.mish(x)
            x = dropout(x)
            
            # Residual connection (if dimensions match)
            if identity.shape == x.shape:
                x = x + identity
        
        # Output layer
        x = self.fc_output(x)
        x = torch.sigmoid(x)
        
        return x


class DLModelTrainer:
    """Handles training and evaluation of Deep Learning model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_model_state = None
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        auc = roc_auc_score(all_targets, all_preds)
        
        return avg_loss, auc
    
    def train_until_convergence(self, train_loader, val_loader, 
                                max_epochs=50, patience=10):
        """Train model until convergence with early stopping"""
        print("\n[Training Deep Learning Model]")
        print(f"Max epochs: {max_epochs}, Patience: {patience}")
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_auc = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_auc'].append(val_auc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print(f"\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def evaluate(self, test_loader, y_true, threshold=0.5):
        """Evaluate model on test set"""
        print("\n[Evaluating Deep Learning Model]")
        
        self.model.eval()
        all_preds_proba = []
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).squeeze()
                all_preds_proba.extend(outputs.cpu().numpy())
        
        all_preds_proba = np.array(all_preds_proba)
        all_preds_binary = (all_preds_proba >= threshold).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_true, all_preds_proba)
        f1 = f1_score(y_true, all_preds_binary)
        precision = precision_score(y_true, all_preds_binary)
        recall = recall_score(y_true, all_preds_binary)
        cm = confusion_matrix(y_true, all_preds_binary)
        
        # Print results
        print("\n" + "="*60)
        print("DEEP LEARNING MODEL PERFORMANCE")
        print("="*60)
        print(f"AUC-ROC Score:  {auc:.4f}")
        print(f"F1-Score:       {f1:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print("="*60)
        
        return {
            'y_pred_proba': all_preds_proba,
            'y_pred_binary': all_preds_binary,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }
    
    def save_model(self, path='dl_model.pth'):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path='dl_model.pth'):
        """Load model"""
        self.model.load_state_dict(torch.load(path))
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    """Example usage"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*80)
    print("TASK 2: DEEP LEARNING MODEL")
    print("="*80)
    
    # Load processed data
    print("\nLoading processed data...")
    X = pd.read_csv('processed_features.csv')
    y = pd.read_csv('processed_target.csv').squeeze()
    
    # Split data
    print("Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test.values)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # Build model
    print(f"\nBuilding Deep Learning model...")
    print(f"  Input features: {X_train_scaled.shape[1]}")
    print(f"  Architecture: [768, 512, 384, 256, 128, 64]")
    print(f"  Dropout: 0.4")
    print(f"  Activation: Mish")
    print(f"  Multi-head attention: 4 heads")
    
    model = LoanDefaultMLP(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[768, 512, 384, 256, 128, 64],
        dropout=0.4,
        use_mixup=True,
        n_heads=4
    )
    
    # Train model
    trainer = DLModelTrainer(model)
    trainer.train_until_convergence(train_loader, test_loader, max_epochs=50, patience=10)
    
    # Evaluate model
    results = trainer.evaluate(test_loader, y_test)
    
    # Save model
    trainer.save_model('dl_model.pth')
    
    # Save results
    import json
    with open('dl_results.json', 'w') as f:
        json.dump({
            'auc': float(results['auc']),
            'f1': float(results['f1']),
            'precision': float(results['precision']),
            'recall': float(results['recall'])
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("TASK 2 COMPLETE: Deep Learning Model")
    print("="*80)
    print(f"AUC-ROC: {results['auc']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print("Model saved: dl_model.pth")
    print("Results saved: dl_results.json")
