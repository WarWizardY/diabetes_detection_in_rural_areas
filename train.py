"""
Diabetes Detection — Random Forest Training Script
====================================================
Trains a Random Forest Classifier to detect diabetes from patient health data.

FEATURES USED (16 total):
  Numeric (14):  chol, stab.glu, hdl, ratio, age, height, weight,
                 bp.1s, bp.1d, waist, hip, time.ppn, BMI*, waist_hip_ratio*
                 (* = engineered features)
  Categorical (2 → one-hot encoded to 4):
                 gender (male/female), location (Buckingham/Louisa), frame (small/medium/large)

  Total after encoding: 18 features

TARGET: diabetes (1 = HbA1c >= 6.5, 0 = not diabetic)

USAGE:
  venv\\Scripts\\python.exe train.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ─── Config ───────────────────────────────────────────────────
DATA_PATH = Path(r"d:\Miniproject\data\diabetes.csv")
OUTPUT_DIR = Path(r"d:\Miniproject\model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("  DIABETES DETECTION — RANDOM FOREST CLASSIFIER")
print("=" * 65)

# ─── 1. LOAD & CLEAN DATA ────────────────────────────────────
print("\n[1/7] Loading and cleaning data...")
df = pd.read_csv(DATA_PATH)

# Create target: diabetes = 1 if glyhb >= 6.5
df['diabetes'] = (df['glyhb'] >= 6.5).astype(int)

# Drop patients without HbA1c (can't label them)
df = df.dropna(subset=['glyhb'])
print(f"  Patients with valid HbA1c: {len(df)}")
print(f"  Diabetic: {df['diabetes'].sum()} | Non-Diabetic: {(df['diabetes'] == 0).sum()}")

# Drop columns we don't need
#   - id: just an identifier
#   - glyhb: this IS the target (would be cheating to use it)
#   - bp.2s, bp.2d: 65% missing, unreliable
df = df.drop(columns=['id', 'glyhb', 'bp.2s', 'bp.2d'])

# ─── 2. FEATURE ENGINEERING ──────────────────────────────────
print("\n[2/7] Engineering features...")

# BMI = weight(lbs) / height(inches)^2 × 703
df['BMI'] = (df['weight'] / (df['height'] ** 2)) * 703
# Waist-to-Hip ratio
df['waist_hip_ratio'] = df['waist'] / df['hip']

print("  + BMI (from height & weight)")
print("  + Waist-to-Hip ratio")

# ─── 3. HANDLE MISSING VALUES & ENCODE CATEGORICALS ──────────
print("\n[3/7] Handling missing values & encoding categoricals...")

# Encode categoricals
df['gender'] = LabelEncoder().fit_transform(df['gender'])           # female=0, male=1
df['location'] = LabelEncoder().fit_transform(df['location'])       # Buckingham=0, Louisa=1
df = pd.get_dummies(df, columns=['frame'], drop_first=False, dtype=int)  # small, medium, large

# Fill remaining missing values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Separate features and target
X = df.drop(columns=['diabetes'])
y = df['diabetes']

feature_names = list(X.columns)
print(f"\n  TOTAL FEATURES: {len(feature_names)}")
print(f"  Features: {feature_names}")

# ─── 4. TRAIN-TEST SPLIT + SMOTE ─────────────────────────────
print("\n[4/7] Splitting data (80/20) and applying SMOTE...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train set: {len(X_train)} samples")
print(f"  Test set:  {len(X_test)} samples")
print(f"  Train class split: {dict(y_train.value_counts())}")

# SMOTE to balance the training set (not test set!)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"  After SMOTE:       {dict(pd.Series(y_train_sm).value_counts())}")

# ─── 5. TRAIN RANDOM FOREST ──────────────────────────────────
print("\n[5/7] Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,          # 200 trees
    max_depth=10,              # prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',   # extra imbalance handling
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)

rf.fit(X_train_sm, y_train_sm)
print("  Model trained!")

# ─── 6. EVALUATE ─────────────────────────────────────────────
print("\n[6/7] Evaluating model...")
print("=" * 65)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"""
  RESULTS ON TEST SET ({len(X_test)} patients)
  ─────────────────────────────────
  Accuracy  : {acc:.2%}
  Precision : {prec:.2%}  (of those we flagged diabetic, how many actually are)
  Recall    : {rec:.2%}  (of actual diabetics, how many we caught)
  F1 Score  : {f1:.2%}  (balance of precision & recall)
  AUC-ROC   : {auc:.4f}  (overall ranking ability, 1.0 = perfect)
""")

print("  CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# 5-Fold Cross Validation
print("  5-FOLD CROSS VALIDATION:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1')
print(f"  F1 scores per fold: {[f'{s:.3f}' for s in cv_scores]}")
print(f"  Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ─── 7. SAVE PLOTS & MODEL ───────────────────────────────────
print(f"\n[7/7] Saving plots and model...")

# --- Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Add text explanation
tn, fp, fn, tp = cm.ravel()
axes[0].text(0.5, -0.15,
    f"True Neg: {tn} | False Pos: {fp} | False Neg: {fn} | True Pos: {tp}",
    transform=axes[0].transAxes, ha='center', fontsize=10, style='italic')

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='#e74c3c', linewidth=2, label=f'Random Forest (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random Guess')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='#e74c3c')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix_roc.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: confusion_matrix_roc.png")

# --- Feature Importance ---
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feat_imp)))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='black', alpha=0.85)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Feature Importance — What Matters Most for Diabetes Detection',
             fontsize=14, fontweight='bold')

# Add value labels
for i, (feat, imp) in enumerate(zip(feat_imp['Feature'], feat_imp['Importance'])):
    ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()
print("  Saved: feature_importance.png")

# --- Save the model ---
joblib.dump(rf, OUTPUT_DIR / 'diabetes_rf_model.pkl')
print("  Saved: diabetes_rf_model.pkl")

# --- Save feature importance as CSV ---
feat_imp_sorted = feat_imp.sort_values('Importance', ascending=False)
feat_imp_sorted.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
print("  Saved: feature_importance.csv")

# ─── FINAL SUMMARY ───────────────────────────────────────────
print("\n" + "=" * 65)
print("  TRAINING COMPLETE!")
print("=" * 65)
print(f"""
  Model: Random Forest (200 trees, max_depth=10)
  Features used: {len(feature_names)}
  Training samples: {len(X_train_sm)} (after SMOTE)
  Test samples: {len(X_test)}
  
  Key metrics:
    Accuracy  = {acc:.2%}
    F1 Score  = {f1:.2%}
    AUC-ROC   = {auc:.4f}
  
  Top 5 most important features:
""")

for i, (_, row) in enumerate(feat_imp_sorted.head(5).iterrows()):
    print(f"    {i+1}. {row['Feature']:20s} → {row['Importance']:.4f}")

print(f"""
  All outputs saved to: {OUTPUT_DIR}
  Model file: diabetes_rf_model.pkl
""")
print("=" * 65)
