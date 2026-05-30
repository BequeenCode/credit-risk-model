# Script to generate sample credit risk dataset
import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 5000

data = {
    'loan_id': np.arange(1, n_samples + 1),
    'age': np.random.normal(45, 15, n_samples).astype(int),
    'income': np.random.normal(60000, 25000, n_samples).astype(int),
    'loan_amount': np.random.normal(150000, 75000, n_samples).astype(int),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
    'interest_rate': np.random.normal(5.5, 2.5, n_samples),
    'employment_years': np.random.normal(8, 5, n_samples).astype(int),
    'num_accounts': np.random.randint(1, 15, n_samples),
    'num_delinquencies': np.random.choice([0, 0, 0, 0, 1, 2, 3], n_samples),
    'credit_score': np.random.normal(650, 100, n_samples).astype(int),
    'employment_type': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples),
    'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
    'loan_purpose': np.random.choice(['Debt Consolidation', 'Home Improvement', 'Business', 'Auto', 'Education'], n_samples),
}

# Target variable with some relationship to features
default_prob = (
    0.1 + 
    0.01 * (data['credit_score'] - 650) / 100 * -1 +
    0.05 * (data['num_delinquencies'] / 3) +
    0.02 * (data['interest_rate'] - 5.5) / 2.5
)
default_prob = np.clip(default_prob, 0.02, 0.3)
data['default'] = (np.random.random(n_samples) < default_prob).astype(int)

df = pd.DataFrame(data)

# Add some missing values (as floats to allow NaN)
df['employment_years'] = df['employment_years'].astype(float)
df['num_delinquencies'] = df['num_delinquencies'].astype(float)
missing_indices = np.random.choice(n_samples, 50, replace=False)
df.loc[missing_indices[:30], 'employment_years'] = np.nan
df.loc[missing_indices[30:], 'num_delinquencies'] = np.nan

# Ensure age and income are positive
df.loc[df['age'] < 18, 'age'] = 18
df.loc[df['income'] < 10000, 'income'] = 10000
df.loc[df['loan_amount'] < 5000, 'loan_amount'] = 5000
df.loc[df['credit_score'] < 300, 'credit_score'] = 300
df.loc[df['credit_score'] > 850, 'credit_score'] = 850
df.loc[df['employment_years'] < 0, 'employment_years'] = 0
df.loc[df['interest_rate'] < 0, 'interest_rate'] = 0.5

df.to_csv('credit_data.csv', index=False)
print(f"Generated credit_data.csv with {len(df)} rows")
