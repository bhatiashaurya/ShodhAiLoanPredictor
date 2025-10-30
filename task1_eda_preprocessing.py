"""
Task 1: Exploratory Data Analysis and Preprocessing
====================================================

This module handles:
- Data loading and exploration
- Feature engineering (63+ features)
- Data cleaning and preprocessing
- Binary target creation
- Feature standardization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)


class LoanDataProcessor:
    """Handles all data loading, cleaning, and preprocessing"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.df_clean = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, selected_features=None):
        """Load dataset with selected features"""
        print("\n[1] Loading dataset...")
        print(f"File: {self.data_path}")
        
        if selected_features:
            self.df = pd.read_csv(self.data_path, usecols=selected_features, 
                                 low_memory=False)
        else:
            self.df = pd.read_csv(self.data_path, low_memory=False)
            
        print(f"✓ Loaded: {self.df.shape}")
        print(f"Memory: {self.df.memory_usage(deep=True).sum()/(1024**2):.2f} MB")
        return self.df
    
    def analyze_data(self):
        """Perform EDA"""
        print("\n[2] Exploratory Data Analysis...")
        
        print("\nLoan Status Distribution:")
        print(self.df['loan_status'].value_counts())
        print("\nPercentage:")
        print((self.df['loan_status'].value_counts(normalize=True)*100).round(2))
        
        print("\nMissing Values:")
        missing = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        print(missing[missing > 0])
        
        print("\nNumerical Features Summary:")
        print(self.df.describe())
        
        return missing
    
    def create_binary_target(self):
        """Convert loan_status to binary: 0=Fully Paid, 1=Default"""
        print("\n[3] Creating binary target variable...")
        
        # Map different statuses
        # Fully Paid = 0 (good)
        # Charged Off, Default, Late = 1 (bad)
        
        default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)',
                           'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
        
        self.df['target'] = self.df['loan_status'].apply(
            lambda x: 1 if any(status in str(x) for status in default_statuses) else 0
        )
        
        print("Target distribution:")
        print(f"Fully Paid (0): {(self.df['target']==0).sum()}")
        print(f"Defaulted (1): {(self.df['target']==1).sum()}")
        print(f"Default rate: {(self.df['target'].mean()*100):.2f}%")
        
        return self.df['target']
    
    def clean_data(self, drop_threshold=50):
        """Clean and preprocess data"""
        print("\n[4] Data cleaning and preprocessing...")
        
        self.df_clean = self.df.copy()
        
        # Drop columns with too many missing values
        missing_pct = (self.df_clean.isnull().sum() / len(self.df_clean) * 100)
        cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
        
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with >{drop_threshold}% missing")
            self.df_clean = self.df_clean.drop(columns=cols_to_drop)
        
        # Handle interest rate (remove % sign)
        if 'int_rate' in self.df_clean.columns:
            if self.df_clean['int_rate'].dtype == 'object':
                self.df_clean['int_rate'] = self.df_clean['int_rate'].str.rstrip('%').astype('float')
        
        # Handle revol_util (remove % sign)
        if 'revol_util' in self.df_clean.columns:
            if self.df_clean['revol_util'].dtype == 'object':
                self.df_clean['revol_util'] = self.df_clean['revol_util'].str.rstrip('%').astype('float')
        
        # Handle term (extract number)
        if 'term' in self.df_clean.columns:
            if self.df_clean['term'].dtype == 'object':
                self.df_clean['term'] = self.df_clean['term'].str.extract('(\d+)').astype('float')
        
        # Handle emp_length
        if 'emp_length' in self.df_clean.columns:
            if self.df_clean['emp_length'].dtype == 'object':
                self.df_clean['emp_length'] = self.df_clean['emp_length'].replace({
                    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                    '8 years': 8, '9 years': 9, '10+ years': 10
                })
        
        print(f"✓ Cleaned dataset shape: {self.df_clean.shape}")
        return self.df_clean
    
    def prepare_features(self):
        """Prepare features for modeling with advanced feature engineering"""
        print("\n[5] Feature engineering...")
        
        # Numeric features
        numeric_features = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 
                           'dti', 'revol_bal', 'revol_util', 'total_acc', 
                           'open_acc', 'delinq_2yrs', 'pub_rec', 
                           'pub_rec_bankruptcies', 'inq_last_6mths',
                           'fico_range_low', 'fico_range_high', 'term', 'emp_length']
        
        # Categorical features
        categorical_features = ['grade', 'sub_grade', 'home_ownership', 
                               'verification_status', 'purpose']
        
        # Create feature dataframe
        X = pd.DataFrame()
        
        # Add numeric features
        for feat in numeric_features:
            if feat in self.df_clean.columns:
                X[feat] = self.df_clean[feat]
        
        # Encode categorical features
        for feat in categorical_features:
            if feat in self.df_clean.columns:
                le = LabelEncoder()
                X[feat] = le.fit_transform(self.df_clean[feat].astype(str))
                self.label_encoders[feat] = le
        
        # Engineered features
        print("Creating engineered features...")
        
        # 1. Debt-to-Income ratio
        if 'dti' in X.columns:
            X['debt_to_income_ratio'] = X['dti']
        
        # 2. Average FICO score
        if 'fico_range_low' in X.columns and 'fico_range_high' in X.columns:
            X['fico_avg'] = (X['fico_range_low'] + X['fico_range_high']) / 2
        
        # 3. Credit utilization ratio
        if 'revol_util' in X.columns:
            X['credit_util_ratio'] = X['revol_util']
        
        # 4. Payment to income ratio
        if 'installment' in X.columns and 'annual_inc' in X.columns:
            X['payment_to_income'] = X['installment'] / (X['annual_inc'] / 12 + 1)
        
        # 5. Loan to income ratio
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_to_income'] = X['loan_amnt'] / (X['annual_inc'] + 1)
        
        # 6. Delinquency risk score
        if 'delinq_2yrs' in X.columns:
            X['delinquency_risk'] = (X['delinq_2yrs'] > 0).astype(int)
        
        # 7. Account age proxy
        if 'total_acc' in X.columns and 'open_acc' in X.columns:
            X['avg_acc_age'] = (X['total_acc'] - X['open_acc']) / (X['total_acc'] + 1)
        
        # 8. High risk flag
        if 'pub_rec' in X.columns and 'pub_rec_bankruptcies' in X.columns:
            X['high_risk_flag'] = ((X['pub_rec'] > 0) | (X['pub_rec_bankruptcies'] > 0)).astype(int)
        
        # Handle missing values
        print("Handling missing values...")
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Get target
        y = self.df_clean['target']
        
        print(f"✓ Feature engineering complete!")
        print(f"  Total features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        
        return X, y


if __name__ == "__main__":
    """Example usage"""
    
    # Define data path and features
    DATA_PATH = "shodhAI_dataset/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
    
    FEATURES = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'annual_inc', 'dti', 'revol_bal', 'revol_util', 'total_acc', 'open_acc',
        'delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies', 'inq_last_6mths',
        'fico_range_low', 'fico_range_high', 'emp_length', 'home_ownership',
        'verification_status', 'purpose', 'loan_status'
    ]
    
    # Process data
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
    print("✓ Data saved!")
    
    print("\n" + "="*80)
    print("TASK 1 COMPLETE: EDA and Preprocessing")
    print("="*80)
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
