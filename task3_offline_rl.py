"""
Task 3: Offline Reinforcement Learning Agent
=============================================

This module implements:
- MDP formulation (State, Action, Reward)
- Q-Network architecture
- Offline RL training with Conservative Q-Learning
- Policy evaluation and expected value calculation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


class OfflineRLDataset(Dataset):
    """
    Dataset for Offline RL
    Converts supervised data to RL transitions: (state, action, reward, next_state, done)
    """
    
    def __init__(self, X, y, loan_amnt, int_rate):
        self.states = torch.FloatTensor(X)
        self.targets = y.values
        self.loan_amnt = loan_amnt.values
        self.int_rate = int_rate.values
        
        # Historical actions (assume all were approved in dataset)
        self.actions = np.ones(len(X))  # 1 = Approve
        
        # Calculate rewards
        self.rewards = self._calculate_rewards()
        
    def _calculate_rewards(self):
        """Calculate reward for each loan based on outcome"""
        rewards = np.zeros(len(self.states))
        
        for i in range(len(self.states)):
            action = self.actions[i]
            outcome = self.targets[i]
            loan_amt = self.loan_amnt[i]
            interest_rate = self.int_rate[i] / 100  # Convert to decimal
            
            if action == 0:  # Deny
                rewards[i] = 0  # No risk, no gain
            else:  # Approve
                if outcome == 0:  # Fully Paid
                    rewards[i] = loan_amt * interest_rate  # Profit from interest
                else:  # Default
                    rewards[i] = -loan_amt  # Loss of principal
        
        return rewards
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx],
            'target': self.targets[idx],
            'loan_amnt': self.loan_amnt[idx],
            'int_rate': self.int_rate[idx]
        }


class SimpleOfflineRLAgent(nn.Module):
    """
    Q-Network for Offline RL
    
    Architecture:
    - Input: State features
    - Hidden layers: [256, 128, 64]
    - Output: Q-values for 2 actions (Deny, Approve)
    """
    
    def __init__(self, state_dim, action_dim=2):
        super(SimpleOfflineRLAgent, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, action_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state):
        """Forward pass to get Q-values"""
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc_out(x)
        return q_values


class OfflineRLTrainer:
    """Handles training and evaluation of Offline RL agent"""
    
    def __init__(self, agent, state_dim, action_dim=2, device='cpu'):
        self.agent = agent.to(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.training_history = {
            'loss': [],
            'q_values': []
        }
    
    def train_agent(self, train_loader, epochs=100, lr=0.0001):
        """
        Train RL agent using Fitted Q-Iteration
        with Conservative Q-Learning (CQL) penalty
        """
        print("\n[Training Offline RL Agent]")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Conservative Q-Learning: Enabled")
        
        optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_q_values = []
            
            for batch in train_loader:
                states = batch['state'].to(self.device)
                actions = batch['action'].long().to(self.device)
                rewards = batch['reward'].float().to(self.device)
                
                # Get Q-values
                q_values = self.agent(states)
                
                # Get Q-value for taken action
                q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                
                # Target: immediate reward (episodic task, no next state)
                target_q = rewards
                
                # TD Loss
                td_loss = F.mse_loss(q_taken, target_q)
                
                # Conservative penalty (CQL): penalize overestimation
                # Subtract logsumexp of Q-values to make them more conservative
                conservative_penalty = torch.logsumexp(q_values, dim=1).mean()
                
                # Total loss
                loss = td_loss + 0.1 * conservative_penalty
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_q_values.append(q_values.mean().item())
            
            # Save history
            avg_loss = epoch_loss / len(train_loader)
            avg_q = np.mean(epoch_q_values)
            self.training_history['loss'].append(avg_loss)
            self.training_history['q_values'].append(avg_q)
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Avg Q-value: {avg_q:.2f}")
        
        print(f"\n✓ Training complete!")
    
    def evaluate_policy(self, test_loader_full):
        """
        Evaluate learned policy on test set
        Calculate Expected Policy Value (profit per loan)
        """
        print("\n[Evaluating Offline RL Agent]")
        
        self.agent.eval()
        
        all_states = []
        all_rewards = []
        all_actions = []
        all_q_deny = []
        all_q_approve = []
        
        with torch.no_grad():
            for batch in test_loader_full:
                states = batch['state'].to(self.device)
                rewards = batch['reward'].cpu().numpy()
                loan_amnt = batch['loan_amnt'].cpu().numpy()
                int_rate = batch['int_rate'].cpu().numpy()
                targets = batch['target'].cpu().numpy()
                
                # Get Q-values
                q_values = self.agent(states).cpu().numpy()
                
                # Choose action (greedy policy)
                actions = np.argmax(q_values, axis=1)
                
                # Calculate actual rewards for chosen actions
                actual_rewards = np.zeros(len(states))
                for i in range(len(states)):
                    if actions[i] == 0:  # Deny
                        actual_rewards[i] = 0
                    else:  # Approve
                        if targets[i] == 0:  # Fully Paid
                            actual_rewards[i] = loan_amnt[i] * (int_rate[i] / 100)
                        else:  # Default
                            actual_rewards[i] = -loan_amnt[i]
                
                all_states.extend(states.cpu().numpy())
                all_rewards.extend(actual_rewards)
                all_actions.extend(actions)
                all_q_deny.extend(q_values[:, 0])
                all_q_approve.extend(q_values[:, 1])
        
        all_rewards = np.array(all_rewards)
        all_actions = np.array(all_actions)
        
        # Calculate metrics
        policy_value = np.mean(all_rewards)
        total_value = np.sum(all_rewards)
        approval_rate = np.mean(all_actions)
        
        # Print results
        print("\n" + "="*60)
        print("OFFLINE RL AGENT PERFORMANCE")
        print("="*60)
        print(f"Expected Policy Value:  ${policy_value:,.2f} per loan")
        print(f"Total Expected Value:   ${total_value:,.2f}")
        print(f"Approval Rate:          {approval_rate:.2%}")
        print(f"Average Q(deny):        {np.mean(all_q_deny):.2f}")
        print(f"Average Q(approve):     {np.mean(all_q_approve):.2f}")
        print("="*60)
        
        return {
            'policy_value': policy_value,
            'total_value': total_value,
            'approval_rate': approval_rate,
            'actions': all_actions,
            'rewards': all_rewards,
            'q_deny': np.array(all_q_deny),
            'q_approve': np.array(all_q_approve)
        }
    
    def save_agent(self, path='rl_agent.pth'):
        """Save agent"""
        torch.save(self.agent.state_dict(), path)
        print(f"✓ Agent saved to {path}")
    
    def load_agent(self, path='rl_agent.pth'):
        """Load agent"""
        self.agent.load_state_dict(torch.load(path))
        print(f"✓ Agent loaded from {path}")


if __name__ == "__main__":
    """Example usage"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    print("\n" + "="*80)
    print("TASK 3: OFFLINE REINFORCEMENT LEARNING")
    print("="*80)
    
    # Load processed data
    print("\nLoading processed data...")
    X = pd.read_csv('processed_features.csv')
    y = pd.read_csv('processed_target.csv').squeeze()
    
    # Need loan_amnt and int_rate for reward calculation
    # Load original cleaned data
    from task1_eda_preprocessing import LoanDataProcessor
    
    DATA_PATH = "shodhAI_dataset/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
    FEATURES = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc', 'open_acc',
        'delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies', 'inq_last_6mths',
        'fico_range_low', 'fico_range_high', 'emp_length', 'home_ownership',
        'verification_status', 'purpose', 'loan_status'
    ]
    
    processor = LoanDataProcessor(DATA_PATH)
    df = processor.load_data(FEATURES)
    processor.create_binary_target()
    processor.clean_data()
    
    # Get loan_amnt and int_rate
    loan_amnt = processor.df_clean['loan_amnt']
    int_rate = processor.df_clean['int_rate']
    
    # Split data
    print("Splitting data (80-20)...")
    X_train, X_test, y_train, y_test, loan_train, loan_test, rate_train, rate_test = train_test_split(
        X, y, loan_amnt, int_rate, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create RL datasets
    print("Creating RL datasets...")
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
    
    agent = SimpleOfflineRLAgent(
        state_dim=X_train_scaled.shape[1],
        action_dim=2
    )
    
    # Train agent
    trainer = OfflineRLTrainer(agent, state_dim=X_train_scaled.shape[1])
    trainer.train_agent(train_rl_loader, epochs=100, lr=0.0001)
    
    # Evaluate agent
    results = trainer.evaluate_policy(test_rl_loader)
    
    # Save agent
    trainer.save_agent('rl_agent.pth')
    
    # Save results
    import json
    with open('rl_results.json', 'w') as f:
        json.dump({
            'policy_value': float(results['policy_value']),
            'total_value': float(results['total_value']),
            'approval_rate': float(results['approval_rate'])
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("TASK 3 COMPLETE: Offline RL Agent")
    print("="*80)
    print(f"Policy Value: ${results['policy_value']:,.2f} per loan")
    print(f"Approval Rate: {results['approval_rate']:.2%}")
    print("Agent saved: rl_agent.pth")
    print("Results saved: rl_results.json")
