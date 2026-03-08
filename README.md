# Diabetes Detection in Rural Areas

A machine learning system for **early diabetes screening using clinical health data**, designed to assist healthcare workers in **resource-limited rural settings**.

The project uses a **Random Forest classifier** trained on patient health metrics to predict diabetes risk and help doctors identify high-risk individuals for further medical testing.

---

# Project Overview

Diabetes is a major global health issue, especially in rural communities where access to advanced diagnostic tests like **HbA1c** may be limited.

This project builds a **machine learning screening tool** that predicts diabetes risk using basic clinical features such as:

* Blood glucose
* Cholesterol
* Blood pressure
* Age
* Body measurements

The model can act as a **first-level screening system** to flag patients who may require further testing.

---

# Dataset

Dataset source:

Kaggle Dataset
[https://www.kaggle.com/datasets/imtkaggleteam/diabetes/data](https://www.kaggle.com/datasets/imtkaggleteam/diabetes/data)

Dataset details:

* **403 patient records**
* **390 valid samples after preprocessing**
* **16.7% diabetic cases**
* **18 features used for prediction**

Example features:

* Stabilized glucose
* Cholesterol
* HDL
* Blood pressure
* Waist circumference
* Age
* Gender
* Body frame size

---

# Exploratory Data Analysis (EDA)

The EDA phase analyzed relationships between health indicators and diabetes risk.

Key observations:

* **Glucose level is the strongest predictor**
* **Age strongly correlates with diabetes risk**
* **Higher cholesterol ratio increases diabetes probability**
* **Older age groups show higher diabetes prevalence**

EDA outputs include:

* Feature distributions
* Correlation heatmaps
* Age vs diabetes risk analysis
* Glucose vs HbA1c relationship

---

# Methodology

### Data Preprocessing

* Removed unnecessary columns
* Handled missing values using median imputation
* Encoded categorical variables
* Created engineered features:

  * **BMI**
  * **Waist-to-Hip Ratio**

### Handling Class Imbalance

The dataset was imbalanced (~1:5 diabetic to non-diabetic).

Solution:

**SMOTE (Synthetic Minority Oversampling Technique)**
applied only to the training data.

### Model Used

Random Forest Classifier

Hyperparameters:

```
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
class_weight = balanced
random_state = 42
```

---

# Model Performance

| Metric    | Score      |
| --------- | ---------- |
| Accuracy  | **92.31%** |
| Precision | 73.33%     |
| Recall    | **84.62%** |
| F1 Score  | 78.57%     |
| AUC-ROC   | **0.956**  |

The model successfully detects most diabetic patients, making it suitable for **screening applications**. 

---

# Feature Importance

Top predictors identified by the Random Forest model:

1. Stabilized Glucose
2. Age
3. Systolic Blood Pressure
4. Cholesterol/HDL Ratio
5. Total Cholesterol

These features align with known **clinical risk factors for Type 2 diabetes**.

---

# Project Structure

```
Miniproject/
│
├── train.py                # Model training pipeline
├── predict.py              # Prediction script
├── eda.py                  # Exploratory data analysis
├── generate_report.py      # Report generation
│
├── eda_outputs/            # Visualizations and EDA results
├── model_outputs/          # Trained models and evaluation outputs
│
├── Diabetes_Detection_Project_Report.pdf
│
├── data/                   # Dataset (not included in repo)
└── venv/                   # Virtual environment (ignored)
```

---

# Installation

Clone the repository:

```
git clone https://github.com/WarWizardY/diabetes_detection_in_rural_areas.git
cd diabetes_detection_in_rural_areas
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Usage

Train the model:

```
python train.py
```

Run predictions:

```
python predict.py
```

Run EDA:

```
python eda.py
```

Generate analysis report:

```
python generate_report.py
```

---

# Applications

This system can help:

* Rural healthcare clinics
* Community health screening programs
* Public health monitoring
* Early detection initiatives

It provides a **low-cost AI-assisted diabetes risk assessment tool**.

---

# Limitations

* Dataset size is relatively small
* Trained on a specific demographic group
* Some medical variables had missing data

Future models could be trained on **larger and more diverse datasets**.

---

# Future Improvements

* Deploy as a **web-based screening tool**
* Train using larger medical datasets
* Compare with models like **XGBoost and SVM**
* Add **Explainable AI (SHAP)** for patient-level explanations
* Integrate with **Electronic Health Record (EHR) systems**

---

# License

This project is released under the **MIT License**.

---

# Author

Yash Patil

Machine Learning | AI | Data Science

GitHub:
[https://github.com/WarWizardY](https://github.com/WarWizardY)

---
