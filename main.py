"""
Main Pipeline: Run All Tasks
=============================

This script runs all 4 tasks in sequence:
1. EDA and Preprocessing
2. Deep Learning Model
3. Offline RL Agent
4. Analysis and Comparison

Usage:
    python main.py
"""

import sys
from datetime import datetime
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Import task modules
from task1_eda_preprocessing import LoanDataProcessor
from task2_deep_learning import LoanDefaultMLP, DLModelTrainer
from task3_offline_rl import OfflineRLDataset, SimpleOfflineRLAgent, OfflineRLTrainer
from task4_analysis import ModelComparison

# Set seeds
np.random.seed(42)
torch.manual_seed(42)


def print_header(task_name):
    """Print formatted task header"""
    print("\n" + "="*80)
    print(f"{task_name}")
    print("="*80)


def main():
    """Main execution pipeline"""
    
    print("\n" + "🚀"*40)
    print("LOAN DEFAULT PREDICTION: DEEP LEARNING vs OFFLINE RL")
    print("🚀"*40)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    DATA_PATH = "shodhAI_dataset/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
    
    FEATURES = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc', 'open_acc',
        'delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies', 'inq_last_6mths',
        'fico_range_low', 'fico_range_high', 'emp_length', 'home_ownership',
        'verification_status', 'purpose', 'loan_status'
    ]
    
    # ========================================================================
    # TASK 1: DATA PROCESSING
    # ========================================================================
    print_header("TASK 1: EXPLORATORY DATA ANALYSIS AND PREPROCESSING")
    
    processor = LoanDataProcessor(DATA_PATH)
    df = processor.load_data(FEATURES)
    processor.analyze_data()
    processor.create_binary_target()
    processor.clean_data()
    X, y = processor.prepare_features()
    
    # Save processed data
    print("\nSaving processed data...")
    X.to_csv('processed_features.csv', index=False)
    y.to_csv('processed_target.csv', index=False)
    print("✓ Processed data saved!")
    
    # Get loan_amnt and int_rate for RL
    loan_amnt = processor.df_clean['loan_amnt']
    int_rate = processor.df_clean['int_rate']
    
    # ========================================================================
    # SPLIT DATA
    # ========================================================================
    print("\n" + "-"*80)
    print("Splitting data (80% train, 20% test)...")
    
    X_train, X_test, y_train, y_test, loan_train, loan_test, rate_train, rate_test = train_test_split(
        X, y, loan_amnt, int_rate, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved!")
    
    # ========================================================================
    # TASK 2: DEEP LEARNING MODEL
    # ========================================================================
    print_header("TASK 2: DEEP LEARNING CLASSIFICATION MODEL")
    
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
    
    # Build Deep Learning model
    print(f"\nBuilding Deep Learning model...")
    print(f"  Input features: {X_train_scaled.shape[1]}")
    print(f"  Architecture: [768, 512, 384, 256, 128, 64]")
    print(f"  Dropout: 0.4")
    print(f"  Activation: Mish")
    print(f"  Multi-head attention: 4 heads")
    
    dl_model = LoanDefaultMLP(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=[768, 512, 384, 256, 128, 64],
        dropout=0.4,
        use_mixup=True,
        n_heads=4
    )
    
    # Train Deep Learning model
    dl_trainer = DLModelTrainer(dl_model)
    dl_trainer.train_until_convergence(train_loader, test_loader, max_epochs=50, patience=10)
    
    # Evaluate Deep Learning model
    dl_results = dl_trainer.evaluate(test_loader, y_test)
    
    # Save Deep Learning model
    dl_trainer.save_model('dl_model.pth')
    
    # ========================================================================
    # TASK 3: OFFLINE REINFORCEMENT LEARNING
    # ========================================================================
    print_header("TASK 3: OFFLINE REINFORCEMENT LEARNING AGENT")
    
    # Create RL datasets
    print("\nCreating RL datasets...")
    train_rl_dataset = OfflineRLDataset(X_train_scaled, y_train, loan_train, rate_train)
    test_rl_dataset = OfflineRLDataset(X_test_scaled, y_test, loan_test, rate_test)
    
    train_rl_loader = DataLoader(train_rl_dataset, batch_size=256, shuffle=True)
    test_rl_loader = DataLoader(test_rl_dataset, batch_size=256)
    
    # Build RL agent
    print(f"\nBuilding RL agent...")
    print(f"  State dim: {X_train_scaled.shape[1]}")
    print(f"  Action dim: 2 (Deny=0, Approve=1)")
    print(f"  Q-Network: [256, 128, 64]")
    print(f"  Conservative penalty: Yes (CQL)")
    
    rl_agent = SimpleOfflineRLAgent(
        state_dim=X_train_scaled.shape[1],
        action_dim=2
    )
    
    # Train RL agent
    rl_trainer = OfflineRLTrainer(rl_agent, state_dim=X_train_scaled.shape[1])
    rl_trainer.train_agent(train_rl_loader, epochs=100, lr=0.0001)
    
    # Evaluate RL agent
    rl_results = rl_trainer.evaluate_policy(test_rl_loader)
    
    # Save RL agent
    rl_trainer.save_agent('rl_agent.pth')
    
    # ========================================================================
    # TASK 4: ANALYSIS AND COMPARISON
    # ========================================================================
    print_header("TASK 4: ANALYSIS AND COMPARISON")
    
    # Create comparison object
    comparison = ModelComparison(dl_results, rl_results, X_test, y_test)
    
    # Generate comprehensive report
    comparison_stats = comparison.generate_report()
    
    # Save all results
    print("\n" + "-"*80)
    print("Saving results...")
    
    # DL results
    with open('dl_results.json', 'w') as f:
        json.dump({
            'auc': float(dl_results['auc']),
            'f1': float(dl_results['f1']),
            'precision': float(dl_results['precision']),
            'recall': float(dl_results['recall'])
        }, f, indent=2)
    
    # RL results
    with open('rl_results.json', 'w') as f:
        json.dump({
            'policy_value': float(rl_results['policy_value']),
            'total_value': float(rl_results['total_value']),
            'approval_rate': float(rl_results['approval_rate'])
        }, f, indent=2)
    
    # Comparison results
    with open('comparison_results.json', 'w') as f:
        json.dump({
            'agreement': float(comparison_stats['agreement']),
            'dl_approval_rate': float(comparison_stats['dl_approval_rate']),
            'rl_approval_rate': float(comparison_stats['rl_approval_rate']),
            'disagreement_cases': int(comparison_stats['disagreement_cases'])
        }, f, indent=2)
    
    print("✓ Results saved!")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n\n" + "🎉"*40)
    print("ALL TASKS COMPLETE!")
    print("🎉"*40)
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\n📊 DEEP LEARNING MODEL:")
    print(f"  AUC-ROC:     {dl_results['auc']:.4f}")
    print(f"  F1-Score:    {dl_results['f1']:.4f}")
    print(f"  Precision:   {dl_results['precision']:.4f}")
    print(f"  Recall:      {dl_results['recall']:.4f}")
    
    print("\n💰 OFFLINE RL AGENT:")
    print(f"  Policy Value:   ${rl_results['policy_value']:,.2f} per loan")
    print(f"  Total Value:    ${rl_results['total_value']:,.2f}")
    print(f"  Approval Rate:  {rl_results['approval_rate']:.2%}")
    
    print("\n🔍 MODEL COMPARISON:")
    print(f"  Agreement:         {comparison_stats['agreement']:.2%}")
    print(f"  Disagreements:     {comparison_stats['disagreement_cases']}")
    print(f"  DL Approval Rate:  {comparison_stats['dl_approval_rate']:.2%}")
    print(f"  RL Approval Rate:  {comparison_stats['rl_approval_rate']:.2%}")
    
    print("\n📁 GENERATED FILES:")
    print("  • processed_features.csv")
    print("  • processed_target.csv")
    print("  • scaler.pkl")
    print("  • dl_model.pth")
    print("  • rl_agent.pth")
    print("  • dl_results.json")
    print("  • rl_results.json")
    print("  • comparison_results.json")
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n✅ Project execution successful!")
    print("✅ All tasks completed and documented!")
    print("✅ Ready for submission!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
