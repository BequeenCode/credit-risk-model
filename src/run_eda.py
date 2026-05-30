"""
Exploratory Data Analysis (EDA) - Credit Risk Model
This script performs comprehensive EDA on the credit risk dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - CREDIT RISK MODEL")
print("="*80)

# Load dataset
print("\n[1/10] Loading dataset...")
df = pd.read_csv('data/raw/credit_data.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Overview
print("\n[2/10] Dataset Overview...")
print(f"\nData Types:\n{df.dtypes}")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumerical Features ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")

# Summary Statistics
print("\n[3/10] Summary Statistics...")
summary_stats = df[numerical_cols].describe().T
summary_stats['skew'] = df[numerical_cols].skew()
summary_stats['kurtosis'] = df[numerical_cols].kurtosis()
print("\nNumerical Features Summary:")
print(summary_stats.round(3))

# Numerical distributions
print("\n[4/10] Creating numerical distributions plot...")
n_cols = len(numerical_cols)
n_rows = (n_cols + 3) // 4  # Calculate rows needed for 4 columns
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
axes = axes.ravel()
for idx, col in enumerate(numerical_cols):
    axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(alpha=0.3)
    skew_val = df[col].skew()
    axes[idx].text(0.98, 0.97, f'Skew: {skew_val:.2f}', 
                   transform=axes[idx].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   ha='right', va='top', fontsize=9)

for idx in range(len(numerical_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('reports/numerical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: numerical_distributions.png")
plt.close()

# Categorical distributions
print("\n[5/10] Creating categorical distributions plot...")
fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 5))
if len(categorical_cols) == 1:
    axes = [axes]

for idx, col in enumerate(categorical_cols):
    counts = df[col].value_counts()
    axes[idx].bar(counts.index, counts.values, color='coral', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(alpha=0.3, axis='y')
    for bar in axes[idx].patches:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('reports/categorical_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: categorical_distributions.png")
plt.close()

# Correlation matrix
print("\n[6/10] Creating correlation matrix...")
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_matrix.png")
plt.close()

# Target correlations
if 'default' in df.columns:
    print("\n[7/10] Analyzing target correlations...")
    target_corr = df[numerical_cols].corrwith(df['default']).sort_values(ascending=False)
    print(f"\nTop Correlations with Default:\n{target_corr.round(3)}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    target_corr.drop('default').sort_values().plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Correlation of Features with Default Target', fontsize=12, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('reports/target_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: target_correlations.png")
    plt.close()

# Missing values
print("\n[8/10] Analyzing missing values...")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_data) > 0:
    print(f"\nMissing Values:\n{missing_data.to_string(index=False)}")
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_by_col.plot(kind='bar', ax=ax, color='salmon')
    ax.set_title('Missing Values by Column', fontsize=12, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/missing_values.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: missing_values.png")
    plt.close()
else:
    print("\n✓ No missing values detected!")

# Outlier detection
print("\n[9/10] Detecting outliers...")
n_cols = len(numerical_cols)
n_rows = (n_cols + 3) // 4  # Calculate rows needed for 4 columns
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
axes = axes.ravel()

outlier_summary = []
for idx, col in enumerate(numerical_cols):
    axes[idx].boxplot(df[col].dropna(), vert=True)
    axes[idx].set_title(f'Box Plot: {col}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(alpha=0.3, axis='y')
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_pct = (len(outliers) / df[col].notna().sum()) * 100
    
    outlier_summary.append({
        'Column': col,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': outlier_pct
    })
    
    axes[idx].text(0.98, 0.97, f'Outliers: {outlier_pct:.1f}%', 
                   transform=axes[idx].transAxes, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                   ha='right', va='top', fontsize=9)

for idx in range(len(numerical_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('reports/outliers_boxplots.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outliers_boxplots.png")
plt.close()

outlier_df = pd.DataFrame(outlier_summary).sort_values('Outlier_Percentage', ascending=False)
print(f"\nOutlier Summary:\n{outlier_df.to_string(index=False)}")

# Target analysis
print("\n[10/10] Analyzing target variable...")
if 'default' in df.columns:
    print(f"\nClass Distribution:\n{df['default'].value_counts()}")
    print(f"\nClass Proportion:\n{df['default'].value_counts(normalize=True).round(3)}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df['default'].value_counts()
    
    axes[0].bar(['Non-Default', 'Default'], counts.values, color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[0].set_title('Target Variable Distribution (Count)', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    axes[1].pie(counts.values, labels=['Non-Default', 'Default'], autopct='%1.1f%%', 
                colors=['green', 'red'], startangle=90, explode=(0.05, 0.05))
    axes[1].set_title('Target Variable Distribution (Proportion)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/target_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: target_distribution.png")
    plt.close()

# Generate insights
print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

insights = []

if 'default' in df.columns:
    target_corr = df[numerical_cols].corrwith(df['default']).abs().sort_values(ascending=False)
    top_feature = target_corr[target_corr.index != 'default'].idxmax()
    top_corr_val = df[top_feature].corr(df['default'])
    insights.append(f"\n1. **Feature Importance**: '{top_feature}' shows the strongest correlation ({top_corr_val:.3f}) with default risk, making it a critical predictor for credit default prediction.")

if df.isnull().sum().sum() > 0:
    missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
    insights.append(f"\n2. **Data Quality**: {len(missing_cols)} column(s) have missing values ({df.isnull().sum().sum()} total missing entries). Recommend using mean/median imputation for numerical features.")
else:
    insights.append(f"\n2. **Data Quality**: No missing values detected. Dataset is complete with high quality.")

if 'default' in df.columns:
    default_rate = (df['default'] == 1).sum() / len(df)
    insights.append(f"\n3. **Class Imbalance**: Default rate is {default_rate*100:.2f}%. {'Significant class imbalance detected - consider SMOTE or class weights.' if default_rate < 0.2 or default_rate > 0.8 else 'Classes appear relatively balanced.'}")

total_outliers = 0
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
    total_outliers += outliers
insights.append(f"\n4. **Outlier Presence**: Detected {total_outliers} outlier instances. Recommend investigating extreme values and considering robust scaling.")

skewed_features = df[numerical_cols].skew().abs()
highly_skewed = skewed_features[skewed_features > 1].index.tolist()
if highly_skewed:
    insights.append(f"\n5. **Skewed Distributions**: {len(highly_skewed)} feature(s) show high skewness. Consider applying log or Box-Cox transformation.")
else:
    insights.append(f"\n5. **Distribution Shape**: Most features show relatively normal distributions. Limited need for transformation.")

for insight in insights:
    print(insight)

print("\n" + "="*80)
print("EDA COMPLETE - All reports saved to reports/")
print("="*80)
