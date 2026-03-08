"""
Diabetes Dataset - Comprehensive Exploratory Data Analysis (EDA)
================================================================
Analyzes the diabetes.csv dataset for:
1. Data shape, types, and missing values
2. Target variable distribution (diabetes classification via glyhb)
3. Descriptive statistics
4. Correlation analysis
5. Feature distributions & outlier detection
6. Key relationships with the target variable
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────
DATA_PATH = Path(r"d:\Miniproject\data\diabetes.csv")
OUTPUT_DIR = Path(r"d:\Miniproject\eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams['figure.dpi'] = 120

# ── 1. Load Data ───────────────────────────────────────────────
print("=" * 70)
print("  DIABETES DATASET — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"\n📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Columns:\n{list(df.columns)}")

# ── 2. Data Types & Info ───────────────────────────────────────
print("\n" + "─" * 70)
print("  DATA TYPES")
print("─" * 70)
print(df.dtypes.to_string())

# ── 3. Missing Values ─────────────────────────────────────────
print("\n" + "─" * 70)
print("  MISSING VALUES")
print("─" * 70)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percent', ascending=False)
if len(missing_df) > 0:
    print(missing_df.to_string())
else:
    print("No missing values found!")

total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(f"\nTotal missing cells: {total_missing} / {total_cells} ({total_missing/total_cells*100:.2f}%)")

# ── 4. Create Diabetes Target Variable ────────────────────────
print("\n" + "─" * 70)
print("  TARGET VARIABLE: DIABETES CLASSIFICATION")
print("─" * 70)

# Create binary diabetes label: glyhb >= 6.5 → diabetic
df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)
# Patients with missing glyhb get NaN for diabetes
df.loc[df['glyhb'].isna(), 'diabetes'] = np.nan

diabetes_counts = df['diabetes'].value_counts(dropna=False)
print("\nDiabetes distribution (glyhb >= 6.5 = diabetic):")
for label, count in diabetes_counts.items():
    if pd.isna(label):
        name = "Missing glyhb"
    elif label == 1:
        name = "Diabetic (1)"
    else:
        name = "Non-Diabetic (0)"
    pct = count / len(df) * 100
    print(f"  {name}: {count} ({pct:.1f}%)")

# ── 5. Descriptive Statistics ─────────────────────────────────
print("\n" + "─" * 70)
print("  DESCRIPTIVE STATISTICS (Numeric Columns)")
print("─" * 70)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove 'id' and 'diabetes' from numeric analysis
analysis_cols = [c for c in numeric_cols if c not in ['id', 'diabetes']]
print(df[analysis_cols].describe().round(2).to_string())

# ── 6. Categorical Variables ──────────────────────────────────
print("\n" + "─" * 70)
print("  CATEGORICAL VARIABLES")
print("─" * 70)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    print(f"\n{col}:")
    vc = df[col].value_counts(dropna=False)
    for val, cnt in vc.items():
        print(f"  {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

# ── 7. Correlation Analysis ───────────────────────────────────
print("\n" + "─" * 70)
print("  CORRELATION WITH glyhb (HbA1c)")
print("─" * 70)
corr_with_target = df[analysis_cols].corrwith(df['glyhb']).dropna().sort_values(ascending=False)
print("\nCorrelation of features with glyhb:")
for feat, corr in corr_with_target.items():
    if feat == 'glyhb':
        continue
    bar = "█" * int(abs(corr) * 30)
    sign = "+" if corr > 0 else "-"
    print(f"  {feat:12s}: {corr:+.4f}  {sign}{bar}")

# ── 8. Correlation Heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
corr_matrix = df[analysis_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='RdBu_r', center=0, square=True, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'correlation_heatmap.png', bbox_inches='tight')
plt.close()
print("\n✅ Saved: correlation_heatmap.png")

# ── 9. Target Distribution Plot ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# glyhb distribution
df['glyhb'].dropna().hist(bins=30, ax=axes[0], color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=6.5, color='red', linestyle='--', linewidth=2, label='Diabetes threshold (6.5)')
axes[0].set_title('Distribution of HbA1c (glyhb)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('HbA1c Value')
axes[0].set_ylabel('Count')
axes[0].legend()

# Diabetes class balance
df_valid = df.dropna(subset=['diabetes'])
diabetes_vc = df_valid['diabetes'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[1].bar(['Non-Diabetic (0)', 'Diabetic (1)'], 
            [diabetes_vc.get(0, 0), diabetes_vc.get(1, 0)],
            color=colors, edgecolor='black', alpha=0.85)
axes[1].set_title('Diabetes Class Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count')
for i, (label, count) in enumerate(zip(['Non-Diabetic', 'Diabetic'], 
                                        [diabetes_vc.get(0, 0), diabetes_vc.get(1, 0)])):
    axes[1].text(i, count + 3, str(count), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_distribution.png', bbox_inches='tight')
plt.close()
print("✅ Saved: target_distribution.png")

# ── 10. Feature Distributions by Diabetes Status ──────────────
key_features = ['stab.glu', 'chol', 'hdl', 'ratio', 'age', 'weight', 'waist', 'hip', 'bp.1s', 'bp.1d']
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    data_0 = df_valid[df_valid['diabetes'] == 0][feat].dropna()
    data_1 = df_valid[df_valid['diabetes'] == 1][feat].dropna()
    axes[i].hist(data_0, bins=20, alpha=0.6, label='Non-Diabetic', color='#2ecc71', edgecolor='black')
    axes[i].hist(data_1, bins=20, alpha=0.6, label='Diabetic', color='#e74c3c', edgecolor='black')
    axes[i].set_title(feat, fontsize=12, fontweight='bold')
    axes[i].legend(fontsize=8)

plt.suptitle('Feature Distributions by Diabetes Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_distributions_by_diabetes.png', bbox_inches='tight')
plt.close()
print("✅ Saved: feature_distributions_by_diabetes.png")

# ── 11. Box Plots for Key Features ────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    df_plot = df_valid[[feat, 'diabetes']].dropna()
    df_plot['diabetes_label'] = df_plot['diabetes'].map({0: 'No', 1: 'Yes'})
    sns.boxplot(data=df_plot, x='diabetes_label', y=feat, ax=axes[i],
                palette={'No': '#2ecc71', 'Yes': '#e74c3c'})
    axes[i].set_title(feat, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Diabetic')

plt.suptitle('Box Plots: Features by Diabetes Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'boxplots_by_diabetes.png', bbox_inches='tight')
plt.close()
print("✅ Saved: boxplots_by_diabetes.png")

# ── 12. Scatter: Glucose vs glyhb ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
scatter_data = df.dropna(subset=['stab.glu', 'glyhb', 'diabetes'])
colors_map = {0: '#2ecc71', 1: '#e74c3c'}
for label, group in scatter_data.groupby('diabetes'):
    ax.scatter(group['stab.glu'], group['glyhb'], 
               c=colors_map[label], label='Diabetic' if label == 1 else 'Non-Diabetic',
               alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
ax.axhline(y=6.5, color='red', linestyle='--', alpha=0.5, label='HbA1c threshold')
ax.set_xlabel('Stabilized Glucose (stab.glu)', fontsize=12)
ax.set_ylabel('Glycosylated Hemoglobin (glyhb)', fontsize=12)
ax.set_title('Glucose vs HbA1c by Diabetes Status', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'glucose_vs_glyhb_scatter.png', bbox_inches='tight')
plt.close()
print("✅ Saved: glucose_vs_glyhb_scatter.png")

# ── 13. Age Distribution by Diabetes ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
age_bins = [0, 30, 40, 50, 60, 70, 100]
age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
df_valid['age_group'] = pd.cut(df_valid['age'], bins=age_bins, labels=age_labels)
age_diabetes = df_valid.groupby(['age_group', 'diabetes']).size().unstack(fill_value=0)
if 1 in age_diabetes.columns:
    age_diabetes['diabetes_rate'] = age_diabetes[1] / (age_diabetes[0] + age_diabetes[1]) * 100
    ax.bar(age_diabetes.index, age_diabetes['diabetes_rate'], color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Diabetes Rate (%)', fontsize=12)
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_title('Diabetes Rate by Age Group', fontsize=14, fontweight='bold')
    for i, (idx, row) in enumerate(age_diabetes.iterrows()):
        total = int(row[0] + row[1])
        ax.text(i, row['diabetes_rate'] + 1, f'{row["diabetes_rate"]:.1f}%\n(n={total})', 
                ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'diabetes_rate_by_age.png', bbox_inches='tight')
plt.close()
print("✅ Saved: diabetes_rate_by_age.png")

# ── 14. Gender Analysis ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
gender_diabetes = df_valid.groupby(['gender', 'diabetes']).size().unstack(fill_value=0)
gender_diabetes.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax.set_title('Diabetes Distribution by Gender', fontsize=14, fontweight='bold')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.legend(['Non-Diabetic', 'Diabetic'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'diabetes_by_gender.png', bbox_inches='tight')
plt.close()
print("✅ Saved: diabetes_by_gender.png")

# ── 15. Outlier Detection ─────────────────────────────────────
print("\n" + "─" * 70)
print("  OUTLIER DETECTION (IQR Method)")
print("─" * 70)
for col in analysis_cols:
    data = df[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data < lower) | (data > upper)]
    if len(outliers) > 0:
        print(f"  {col:12s}: {len(outliers):3d} outliers ({len(outliers)/len(data)*100:.1f}%) "
              f"[Range: {lower:.1f} – {upper:.1f}]")

# ── 16. Pair Plot of Top Features ─────────────────────────────
top_features = ['stab.glu', 'ratio', 'age', 'chol', 'waist']
pair_df = df_valid[top_features + ['diabetes']].dropna()
pair_df['diabetes_label'] = pair_df['diabetes'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
g = sns.pairplot(pair_df, hue='diabetes_label', 
                 palette={'Non-Diabetic': '#2ecc71', 'Diabetic': '#e74c3c'},
                 diag_kind='hist', plot_kws={'alpha': 0.5, 'edgecolor': 'black', 'linewidth': 0.3},
                 height=2.5)
g.fig.suptitle('Pair Plot: Top Predictive Features', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(OUTPUT_DIR / 'pairplot_top_features.png', bbox_inches='tight')
plt.close()
print("✅ Saved: pairplot_top_features.png")

# ── 17. Summary & Recommendations ─────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY & KEY FINDINGS")
print("=" * 70)

valid_count = df_valid.shape[0]
diabetic_count = int(df_valid['diabetes'].sum())
non_diabetic_count = valid_count - diabetic_count

print(f"""
📊 DATASET OVERVIEW:
   • Total patients: {df.shape[0]}
   • Patients with valid HbA1c: {valid_count}
   • Diabetic (glyhb ≥ 6.5): {diabetic_count} ({diabetic_count/valid_count*100:.1f}%)
   • Non-Diabetic: {non_diabetic_count} ({non_diabetic_count/valid_count*100:.1f}%)
   • Class imbalance ratio: ~1:{non_diabetic_count//diabetic_count} (Diabetic:Non-Diabetic)

🔬 KEY CORRELATIONS WITH HbA1c:
   Top positive correlations →  stab.glu (glucose), ratio, age, chol, waist
   These are likely the STRONGEST PREDICTORS of diabetes.

⚠️  CLASS IMBALANCE:
   Only ~{diabetic_count/valid_count*100:.0f}% of patients are diabetic.
   → Need SMOTE, class weights, or stratified sampling.

📁 MISSING DATA:
   bp.2s, bp.2d have ~{missing_pct.get('bp.2s', 0):.0f}% missing (only measured on some patients)
   Some patients missing height, weight, frame, glyhb.

🎯 RECOMMENDED NEXT STEPS:
   1. Clean data: handle missing values (imputation or removal)
   2. Encode categoricals: gender, location, frame
   3. Feature engineering: BMI from height/weight, waist-hip ratio
   4. Train classification models with cross-validation
   5. Use SMOTE or class_weight='balanced' for imbalance
""")

print("✅ EDA Complete! All plots saved to:", OUTPUT_DIR)
print("=" * 70)
