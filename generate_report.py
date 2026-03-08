"""
Diabetes Detection — Professional Project Report (PDF Generator)
=================================================================
Generates a structured, technical PDF report with:
  - Title page
  - Abstract
  - Introduction & Problem Statement
  - Dataset Description
  - EDA Findings with embedded graphs
  - Methodology (preprocessing, SMOTE, Random Forest)
  - Results & Evaluation with metrics and plots
  - Feature Importance Analysis
  - Conclusion & Future Scope

USAGE:  venv\Scripts\python.exe generate_report.py
"""

from fpdf import FPDF
from pathlib import Path
import pandas as pd
import os

# ─── Paths ────────────────────────────────────────────────────
EDA_DIR = Path(r"d:\Miniproject\eda_outputs")
MODEL_DIR = Path(r"d:\Miniproject\model_outputs")
OUTPUT_PDF = Path(r"d:\Miniproject\Diabetes_Detection_Project_Report.pdf")


class ReportPDF(FPDF):
    """Custom PDF class with header/footer and helper methods."""

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Diabetes Detection Using Random Forest Classification", align="L")
            self.cell(0, 8, f"Page {self.page_no()}", align="R")
            self.ln(12)
            # Divider line
            self.set_draw_color(41, 128, 185)
            self.set_line_width(0.5)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Mini Project Report  |  Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        """Major section heading."""
        self.ln(6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        # Underline
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def sub_title(self, title):
        """Sub-section heading."""
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        """Normal paragraph text."""
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6.5, text)
        self.ln(2)

    def bullet(self, text, indent=15):
        """Bullet point."""
        left_margin = self.l_margin
        # Arrow marker
        self.set_x(left_margin + indent)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(41, 128, 185)
        self.cell(5, 6.5, ">")
        # Text
        self.set_x(left_margin + indent + 7)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        text_width = self.w - self.r_margin - (left_margin + indent + 7)
        self.multi_cell(text_width, 6.5, text)

    def key_value(self, key, value, indent=15):
        """Key: value pair."""
        left_margin = self.l_margin
        self.set_x(left_margin + indent)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(44, 62, 80)
        key_w = self.get_string_width(key + ": ") + 2
        self.cell(key_w, 6.5, f"{key}: ")
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        text_width = self.w - self.r_margin - self.get_x()
        self.multi_cell(text_width, 6.5, str(value))

    def add_image_centered(self, img_path, w=170, caption=None):
        """Add an image centered with optional caption."""
        if not os.path.exists(img_path):
            self.body_text(f"[Image not found: {img_path}]")
            return
        x = (210 - w) / 2
        # Check if enough space, otherwise new page
        if self.get_y() + 90 > 270:
            self.add_page()
        self.image(str(img_path), x=x, w=w)
        if caption:
            self.ln(2)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def colored_box(self, title, text, color_rgb=(41, 128, 185)):
        """Highlighted info box."""
        r, g, b = color_rgb
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_fill_color(235, 245, 255)
        self.set_text_color(50, 50, 50)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text, fill=True)
        self.ln(4)

    def metric_table(self, metrics_dict):
        """Render a metrics table."""
        self.set_font("Helvetica", "B", 11)
        col_w = 60
        val_w = 50
        # Header
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(col_w, 9, "  Metric", fill=True, border=1)
        self.cell(val_w, 9, "  Score", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
        # Rows
        self.set_font("Helvetica", "", 11)
        self.set_text_color(50, 50, 50)
        fill = False
        for k, v in metrics_dict.items():
            if fill:
                self.set_fill_color(240, 248, 255)
            else:
                self.set_fill_color(255, 255, 255)
            self.cell(col_w, 8, f"  {k}", fill=True, border=1)
            self.cell(val_w, 8, f"  {v}", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
            fill = not fill
        self.ln(4)


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ══════════════════════════════════════════════════════════
    # TITLE PAGE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.ln(40)

    # Top decorative bar
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.rect(0, 8, 210, 3, "F")
    pdf.set_fill_color(52, 152, 219)
    pdf.rect(0, 8, 210, 2, "F")

    # Title
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 15, "Diabetes Detection", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 12, "Using Random Forest Classification", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    # Decorative line
    pdf.set_draw_color(41, 128, 185)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "A Machine Learning Approach to Early Diabetes Screening", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "in Rural Healthcare Settings", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(20)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Mini Project Report", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Technology: Python | Scikit-Learn | Random Forest | SMOTE", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Dataset: Rural African-American Diabetes Study (403 patients)", align="C", new_x="LMARGIN", new_y="NEXT")

    # Bottom bar
    pdf.set_fill_color(41, 128, 185)
    pdf.rect(0, 285, 210, 12, "F")
    pdf.set_fill_color(52, 152, 219)
    pdf.rect(0, 283, 210, 2, "F")

    # ══════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(41, 128, 185)
    pdf.set_line_width(0.8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    toc_items = [
        ("1", "Abstract", ""),
        ("2", "Introduction & Problem Statement", ""),
        ("3", "Dataset Description", ""),
        ("4", "Exploratory Data Analysis (EDA)", ""),
        ("  4.1", "Target Variable Distribution", ""),
        ("  4.2", "Correlation Analysis", ""),
        ("  4.3", "Feature Distributions by Diabetes Status", ""),
        ("  4.4", "Diabetes Rate by Age Group", ""),
        ("  4.5", "Outlier Analysis", ""),
        ("5", "Methodology", ""),
        ("  5.1", "Data Preprocessing", ""),
        ("  5.2", "Feature Engineering", ""),
        ("  5.3", "Handling Class Imbalance (SMOTE)", ""),
        ("  5.4", "Random Forest Algorithm", ""),
        ("6", "Results & Evaluation", ""),
        ("  6.1", "Performance Metrics", ""),
        ("  6.2", "Confusion Matrix", ""),
        ("  6.3", "ROC Curve Analysis", ""),
        ("7", "Feature Importance Analysis", ""),
        ("8", "Conclusion & Future Scope", ""),
    ]

    for num, title, page in toc_items:
        is_sub = num.startswith("  ")
        pdf.set_font("Helvetica", "" if is_sub else "B", 11 if is_sub else 12)
        pdf.set_text_color(80, 80, 80) if is_sub else pdf.set_text_color(44, 62, 80)
        indent = 15 if is_sub else 0
        pdf.cell(indent, 8, "")
        pdf.cell(0, 8, f"{num.strip()}.  {title}", new_x="LMARGIN", new_y="NEXT")

    # ══════════════════════════════════════════════════════════
    # 1. ABSTRACT
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("1", "Abstract")
    pdf.body_text(
        "Diabetes mellitus is one of the most prevalent chronic diseases globally, affecting millions of "
        "individuals, particularly in underserved rural communities where access to advanced diagnostic "
        "tools is limited. Early detection is critical for preventing severe complications such as "
        "cardiovascular disease, kidney failure, and vision loss."
    )
    pdf.body_text(
        "This project develops a machine learning-based diabetes detection system using a Random Forest "
        "Classifier trained on clinical and demographic data from 403 rural African-American patients. "
        "The model analyzes 18 features including blood glucose levels, cholesterol markers, blood "
        "pressure, body measurements, and engineered features (BMI, waist-to-hip ratio) to predict "
        "whether a patient is diabetic based on the HbA1c threshold of 6.5%."
    )
    pdf.body_text(
        "The trained model achieved an AUC-ROC score of 0.956, demonstrating excellent discriminative "
        "ability. The system identified stabilized glucose, patient age, and systolic blood pressure as "
        "the top three predictors of diabetes. This tool can serve as a rapid screening aid for "
        "healthcare providers in resource-limited settings, enabling early intervention and improved "
        "patient outcomes."
    )

    # Highlight box
    pdf.colored_box(
        "Key Results",
        "  Model: Random Forest (200 trees)  |  AUC-ROC: 0.956  |  Features: 18\n"
        "  Top Predictors: Glucose (31.5%), Age (17.2%), Systolic BP (8.7%)\n"
        "  Application: Rapid diabetes screening for rural healthcare clinics"
    )

    # ══════════════════════════════════════════════════════════
    # 2. INTRODUCTION & PROBLEM STATEMENT
    # ══════════════════════════════════════════════════════════
    pdf.section_title("2", "Introduction & Problem Statement")

    pdf.sub_title("2.1 Background")
    pdf.body_text(
        "Diabetes mellitus is a metabolic disorder characterized by elevated blood sugar levels over "
        "a prolonged period. According to the World Health Organization, approximately 422 million "
        "people worldwide have diabetes, and 1.5 million deaths are directly attributed to it annually. "
        "Type 2 diabetes accounts for about 90% of all cases and is strongly linked to obesity, "
        "physical inactivity, and poor dietary habits."
    )
    pdf.body_text(
        "The gold standard for diabetes diagnosis is the Hemoglobin A1c (HbA1c) test, which measures "
        "average blood glucose levels over the past 2-3 months. A value of 6.5% or higher indicates "
        "diabetes. However, this specialized laboratory test is not always readily available in rural "
        "or low-resource healthcare settings."
    )

    pdf.sub_title("2.2 Problem Statement")
    pdf.body_text(
        "In rural healthcare settings, particularly in underserved African-American communities, "
        "patients may not have easy access to HbA1c testing. The challenge is:"
    )
    pdf.ln(2)
    pdf.colored_box(
        "Problem",
        '  "Can we predict whether a patient has diabetes using only basic health checkup\n'
        '   data (blood glucose, cholesterol, blood pressure, body measurements, and\n'
        '   demographics) so that doctors can quickly flag high-risk patients for further testing?"',
        (231, 76, 60)
    )

    pdf.sub_title("2.3 How This Helps")
    pdf.bullet("Doctors: Quickly identify high-risk patients during routine checkups, without waiting for specialized lab results.")
    pdf.bullet("Patients: Early detection leads to early treatment, preventing severe complications like kidney disease, heart attacks, and blindness.")
    pdf.bullet("Rural Clinics: Acts as a cost-effective screening tool where advanced diagnostics are unavailable or expensive.")
    pdf.bullet("Public Health: Enables large-scale screening programs for diabetes in vulnerable populations.")

    pdf.sub_title("2.4 Objective")
    pdf.body_text(
        "To train a Random Forest classification model on clinical data from 403 patients that can "
        "accurately detect diabetes and identify the most important predictive features, providing an "
        "interpretable and deployable screening solution."
    )

    # ══════════════════════════════════════════════════════════
    # 3. DATASET DESCRIPTION
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("3", "Dataset Description")

    pdf.sub_title("3.1 Data Source")
    pdf.body_text(
        "The dataset consists of clinical records from 403 rural African-American patients. It includes "
        "demographic information, laboratory test results, body measurements, and blood pressure readings. "
        "Each patient's diabetes status was determined by their HbA1c (glyhb) value: patients with "
        "HbA1c >= 6.5 were classified as diabetic."
    )

    pdf.sub_title("3.2 Feature Description")
    # Feature table
    features = [
        ("chol", "Total Cholesterol", "mg/dL", "Numeric"),
        ("stab.glu", "Stabilized Glucose", "mg/dL", "Numeric"),
        ("hdl", "HDL (Good) Cholesterol", "mg/dL", "Numeric"),
        ("ratio", "Cholesterol/HDL Ratio", "-", "Numeric"),
        ("age", "Patient Age", "years", "Numeric"),
        ("gender", "Gender", "M/F", "Categorical"),
        ("height", "Height", "inches", "Numeric"),
        ("weight", "Weight", "pounds", "Numeric"),
        ("frame", "Body Frame Size", "S/M/L", "Categorical"),
        ("bp.1s", "Systolic Blood Pressure", "mmHg", "Numeric"),
        ("bp.1d", "Diastolic Blood Pressure", "mmHg", "Numeric"),
        ("waist", "Waist Circumference", "inches", "Numeric"),
        ("hip", "Hip Circumference", "inches", "Numeric"),
        ("location", "Clinic Location", "-", "Categorical"),
        ("time.ppn", "Time Since Last Meal", "minutes", "Numeric"),
        ("glyhb", "Glyc. Hemoglobin (HbA1c)", "%", "TARGET"),
    ]

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(28, 8, "  Feature", fill=True, border=1)
    pdf.cell(60, 8, "  Description", fill=True, border=1)
    pdf.cell(25, 8, "  Unit", fill=True, border=1)
    pdf.cell(25, 8, "  Type", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    fill = False
    for feat, desc, unit, ftype in features:
        if fill:
            pdf.set_fill_color(240, 248, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        if ftype == "TARGET":
            pdf.set_fill_color(255, 235, 235)
            pdf.set_font("Helvetica", "B", 9)
        pdf.cell(28, 7, f"  {feat}", fill=True, border=1)
        pdf.cell(60, 7, f"  {desc}", fill=True, border=1)
        pdf.cell(25, 7, f"  {unit}", fill=True, border=1)
        pdf.cell(25, 7, f"  {ftype}", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        fill = not fill
    pdf.ln(4)

    pdf.sub_title("3.3 Dataset Statistics")
    pdf.key_value("Total Records", "403 patients")
    pdf.key_value("Valid Records", "390 (13 excluded due to missing HbA1c)")
    pdf.key_value("Diabetic Patients", "65 (16.7%)")
    pdf.key_value("Non-Diabetic Patients", "325 (83.3%)")
    pdf.key_value("Class Ratio", "1:5 (imbalanced dataset)")
    pdf.key_value("Missing Data Rate", "7.51% of all cells")

    pdf.sub_title("3.4 Missing Data Analysis")
    pdf.body_text(
        "Several columns contained missing values. The most significant were bp.2s and bp.2d "
        "(second blood pressure readings) with 65% missing, as these were only measured on a subset "
        "of patients. These columns were dropped from the analysis. Remaining missing values were "
        "imputed using the median strategy."
    )

    # ══════════════════════════════════════════════════════════
    # 4. EXPLORATORY DATA ANALYSIS
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("4", "Exploratory Data Analysis (EDA)")
    pdf.body_text(
        "Exploratory Data Analysis was performed to understand the distribution of features, uncover "
        "relationships between variables, and identify patterns distinguishing diabetic from non-diabetic patients."
    )

    pdf.sub_title("4.1 Target Variable Distribution")
    pdf.body_text(
        "The target variable (diabetes) was derived from HbA1c values. The distribution shows a clear "
        "right skew in HbA1c values, with most patients clustered below 6.5. The dataset is imbalanced "
        "with only 16.7% diabetic cases."
    )
    pdf.add_image_centered(EDA_DIR / "target_distribution.png", w=170,
                            caption="Figure 1: Distribution of HbA1c values and diabetes class balance")

    pdf.sub_title("4.2 Correlation Analysis")
    pdf.body_text(
        "Pearson correlation analysis reveals the strength and direction of linear relationships "
        "between features and the target variable (HbA1c). Key findings:"
    )
    pdf.bullet("Stabilized glucose (stab.glu) has the strongest correlation with HbA1c (r = 0.749), making it the single most powerful predictor.")
    pdf.bullet("Age shows moderate positive correlation (r = 0.339), indicating higher diabetes risk with advancing age.")
    pdf.bullet("Cholesterol/HDL ratio (r = 0.329) and total cholesterol (r = 0.247) are also significant predictors.")
    pdf.bullet("HDL cholesterol shows a weak negative correlation (r = -0.149), suggesting higher HDL is protective against diabetes.")
    pdf.ln(2)

    pdf.add_image_centered(EDA_DIR / "correlation_heatmap.png", w=165,
                            caption="Figure 2: Correlation heatmap showing inter-feature relationships")

    pdf.add_page()
    pdf.sub_title("4.3 Feature Distributions by Diabetes Status")
    pdf.body_text(
        "Comparing feature distributions between diabetic and non-diabetic patients reveals clear "
        "separability in several features. Diabetic patients consistently show higher glucose levels, "
        "cholesterol ratios, and waist circumferences. The box plots further highlight the median "
        "differences and outlier patterns."
    )
    pdf.add_image_centered(EDA_DIR / "feature_distributions_by_diabetes.png", w=180,
                            caption="Figure 3: Histograms of key features split by diabetes status")
    pdf.add_image_centered(EDA_DIR / "boxplots_by_diabetes.png", w=180,
                            caption="Figure 4: Box plots comparing feature distributions between groups")

    pdf.add_page()
    pdf.sub_title("4.4 Glucose vs HbA1c Relationship")
    pdf.body_text(
        "The scatter plot below demonstrates the strong linear relationship between stabilized glucose "
        "and HbA1c. Most diabetic patients (red) cluster in the upper-right region with high glucose "
        "and high HbA1c, while non-diabetic patients cluster in the lower-left."
    )
    pdf.add_image_centered(EDA_DIR / "glucose_vs_glyhb_scatter.png", w=145,
                            caption="Figure 5: Scatter plot of stabilized glucose vs HbA1c")

    pdf.sub_title("4.5 Diabetes Rate by Age Group")
    pdf.body_text(
        "Age is a well-known risk factor for Type 2 diabetes. The chart below confirms this pattern "
        "in our dataset, with diabetes prevalence increasing dramatically from ~5% in patients under 30 "
        "to over 25% in patients aged 70+."
    )
    pdf.add_image_centered(EDA_DIR / "diabetes_rate_by_age.png", w=145,
                            caption="Figure 6: Diabetes prevalence increases sharply with age")

    pdf.sub_title("4.6 Key EDA Takeaways")
    pdf.bullet("Glucose is the dominant predictor - any screening tool must include glucose testing.")
    pdf.bullet("Age, cholesterol ratio, and waist circumference are important secondary predictors.")
    pdf.bullet("The dataset is imbalanced (1:5) - requires careful handling during model training.")
    pdf.bullet("Blood pressure readings (bp.2s, bp.2d) have too much missing data to be useful.")
    pdf.bullet("No strong gender or location bias was observed in diabetes prevalence.")

    # ══════════════════════════════════════════════════════════
    # 5. METHODOLOGY
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("5", "Methodology")

    pdf.sub_title("5.1 Data Preprocessing Pipeline")
    pdf.body_text("The raw data underwent a systematic cleaning and transformation pipeline:")
    pdf.ln(2)
    pdf.bullet("Step 1 - Target Creation: Binary diabetes label created from HbA1c (>= 6.5 = diabetic). Patients without HbA1c values were excluded.")
    pdf.bullet("Step 2 - Column Removal: Dropped id (identifier), glyhb (target source -- using it would be data leakage), bp.2s and bp.2d (65% missing).")
    pdf.bullet("Step 3 - Missing Value Imputation: Remaining missing values filled with the column median. Median was chosen over mean to be robust against outliers.")
    pdf.bullet("Step 4 - Categorical Encoding: Gender encoded as binary (0/1), location as binary, body frame one-hot encoded into 3 columns (small, medium, large).")

    pdf.sub_title("5.2 Feature Engineering")
    pdf.body_text("Two additional features were engineered from existing measurements to provide clinically meaningful metrics:")
    pdf.ln(2)
    pdf.bullet("BMI (Body Mass Index): Calculated as weight(lbs) / height(inches)^2 x 703. BMI is a widely used indicator of obesity, which is a major diabetes risk factor.")
    pdf.bullet("Waist-to-Hip Ratio: Calculated as waist / hip circumference. This ratio is a strong indicator of abdominal obesity, which is closely linked to insulin resistance and Type 2 diabetes.")
    pdf.ln(2)
    pdf.body_text("After preprocessing, the final feature set consisted of 18 features (14 numeric + 4 encoded categorical).")

    pdf.sub_title("5.3 Handling Class Imbalance with SMOTE")
    pdf.body_text(
        "The dataset has a significant class imbalance: only 65 diabetic patients (16.7%) versus 325 "
        "non-diabetic (83.3%). Training a model directly on this data would bias it toward always "
        "predicting \"non-diabetic\" since that would already achieve 83% accuracy."
    )
    pdf.body_text(
        "To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the "
        "training set only. SMOTE works by creating synthetic samples of the minority class (diabetic) "
        "by interpolating between existing diabetic samples in feature space. This balances the classes "
        "without simply duplicating existing records."
    )
    pdf.colored_box(
        "Important: SMOTE Applied to Training Set Only",
        "  SMOTE was applied ONLY to the training data (80%), NOT the test data (20%).\n"
        "  This prevents data leakage and ensures the model is evaluated on real,\n"
        "  unmodified patient data, giving honest performance estimates.",
        (46, 204, 113)
    )

    pdf.sub_title("5.4 Random Forest Classifier")
    pdf.body_text(
        "Random Forest is an ensemble learning algorithm that operates by constructing multiple "
        "decision trees during training and outputting the majority vote of the individual trees. "
        "It was chosen for this project because:"
    )
    pdf.bullet("It handles both numerical and categorical features naturally.")
    pdf.bullet("It is robust against overfitting due to the bagging mechanism.")
    pdf.bullet("It provides built-in feature importance rankings, which are clinically valuable.")
    pdf.bullet("It performs well on small datasets with moderate dimensionality.")
    pdf.bullet("It requires minimal hyperparameter tuning compared to other algorithms.")

    pdf.ln(2)
    pdf.body_text("Model hyperparameters used:")
    pdf.key_value("n_estimators", "200 (number of decision trees)")
    pdf.key_value("max_depth", "10 (prevents overfitting on small dataset)")
    pdf.key_value("min_samples_split", "5 (minimum samples to split a node)")
    pdf.key_value("min_samples_leaf", "2 (minimum samples in a leaf)")
    pdf.key_value("class_weight", "balanced (additional imbalance handling)")
    pdf.key_value("random_state", "42 (reproducibility)")

    pdf.ln(2)
    pdf.body_text(
        "The train-test split was 80/20 with stratification to maintain the class ratio in both sets. "
        "5-fold stratified cross-validation was also performed for robust evaluation."
    )

    # ══════════════════════════════════════════════════════════
    # 6. RESULTS & EVALUATION
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("6", "Results & Evaluation")

    pdf.sub_title("6.1 Performance Metrics")
    pdf.body_text(
        "The model was evaluated on the held-out test set of 78 patients (20% of data). "
        "The following metrics were computed:"
    )

    pdf.metric_table({
        "Accuracy": "92.31%",
        "Precision": "73.33%",
        "Recall (Sensitivity)": "84.62%",
        "F1 Score": "78.57%",
        "AUC-ROC": "0.9562",
    })

    pdf.body_text("Interpretation of each metric:")
    pdf.bullet("Accuracy (92.31%): Overall, the model correctly classifies 92 out of 100 patients.")
    pdf.bullet("Precision (73.33%): When the model flags a patient as diabetic, it is correct about 73% of the time. The 27% false positives would receive unnecessary follow-up, but this is acceptable for a screening tool.")
    pdf.bullet("Recall/Sensitivity (84.62%): The model catches 85% of actual diabetic patients. This is critical -- missing a diabetic patient (false negative) has serious consequences.")
    pdf.bullet("F1 Score (78.57%): The harmonic mean of precision and recall, providing a balanced measure of performance.")
    pdf.bullet("AUC-ROC (0.956): The model has excellent discriminative ability. A score of 1.0 would be perfect, and 0.5 would be random guessing.")

    pdf.sub_title("6.2 Confusion Matrix & ROC Curve")
    pdf.body_text(
        "The confusion matrix and ROC curve provide visual evidence of the model's performance:"
    )
    pdf.add_image_centered(MODEL_DIR / "confusion_matrix_roc.png", w=180,
                            caption="Figure 7: Confusion Matrix and ROC Curve for the Random Forest model")

    pdf.body_text("Confusion Matrix Breakdown (on 78 test patients):")
    pdf.bullet("True Negatives (61): Non-diabetic patients correctly identified as non-diabetic.")
    pdf.bullet("True Positives (11): Diabetic patients correctly detected by the model.")
    pdf.bullet("False Positives (4): Non-diabetic patients incorrectly flagged as diabetic (would get extra testing).")
    pdf.bullet("False Negatives (2): Diabetic patients missed by the model (most concerning).")

    pdf.ln(2)
    pdf.colored_box(
        "Clinical Significance",
        "  Out of 13 actual diabetic patients in the test set, the model correctly\n"
        "  identified 11 (84.6%). Only 2 diabetic patients were missed. For a screening\n"
        "  tool, this high sensitivity is critical for patient safety.",
        (155, 89, 182)
    )

    # ══════════════════════════════════════════════════════════
    # 7. FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("7", "Feature Importance Analysis")
    pdf.body_text(
        "One of the key advantages of Random Forest is its ability to rank features by importance. "
        "Feature importance is calculated based on how much each feature reduces impurity (Gini impurity) "
        "across all decision trees in the forest."
    )

    pdf.add_image_centered(MODEL_DIR / "feature_importance.png", w=155,
                            caption="Figure 8: Feature importance ranking from the trained Random Forest model")

    pdf.sub_title("7.1 Top Predictive Features")

    # Load feature importance
    feat_imp = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    pdf.body_text("The five most important features for diabetes detection are:")
    pdf.ln(2)

    top_5 = feat_imp.head(5)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(41, 128, 185)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(10, 8, "  #", fill=True, border=1)
    pdf.cell(45, 8, "  Feature", fill=True, border=1)
    pdf.cell(30, 8, "  Importance", fill=True, border=1)
    pdf.cell(95, 8, "  Clinical Significance", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")

    explanations = [
        "Directly measures blood sugar, the primary marker for diabetes",
        "Type 2 diabetes risk increases significantly with age",
        "Elevated systolic BP is common in diabetic patients",
        "Higher chol/HDL ratio indicates metabolic dysfunction",
        "Total cholesterol is linked to metabolic syndrome",
    ]

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    for i, (_, row) in enumerate(top_5.iterrows()):
        fill_color = (240, 248, 255) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        pdf.cell(10, 7, f"  {i+1}", fill=True, border=1)
        pdf.cell(45, 7, f"  {row['Feature']}", fill=True, border=1)
        pdf.cell(30, 7, f"  {row['Importance']:.4f}", fill=True, border=1)
        pdf.cell(95, 7, f"  {explanations[i]}", fill=True, border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.sub_title("7.2 Insights from Feature Importance")
    pdf.bullet("Glucose dominance (31.5%): Stabilized glucose alone accounts for nearly a third of the model's predictive power. This confirms that glucose testing is essential for diabetes screening.")
    pdf.bullet("Age matters (17.2%): After glucose, age is the second most important feature. This aligns with medical knowledge that Type 2 diabetes risk increases with age.")
    pdf.bullet("Blood pressure (8.7%): Hypertension and diabetes are closely linked comorbidities. Systolic BP contributes significantly to predictions.")
    pdf.bullet("Body composition features (BMI, waist, waist-hip ratio): Together, these contribute ~12.7%, highlighting the role of obesity in diabetes risk.")
    pdf.bullet("Demographics matter less: Gender (0.4%) and location (0.7%) have minimal predictive power, suggesting diabetes risk is primarily driven by clinical and anthropometric factors, not demographics.")

    # ══════════════════════════════════════════════════════════
    # 8. CONCLUSION & FUTURE SCOPE
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_title("8", "Conclusion & Future Scope")

    pdf.sub_title("8.1 Conclusion")
    pdf.body_text(
        "This project successfully developed a machine learning-based diabetes detection system "
        "using a Random Forest Classifier. The key findings and achievements are:"
    )
    pdf.bullet("The model achieves 92.31% accuracy and an AUC-ROC of 0.956 on unseen test data, demonstrating strong predictive capability.")
    pdf.bullet("Stabilized glucose (31.5%), age (17.2%), and systolic blood pressure (8.7%) were identified as the three strongest predictors of diabetes.")
    pdf.bullet("The SMOTE technique effectively addressed the class imbalance problem, enabling the model to detect 84.6% of actual diabetic cases.")
    pdf.bullet("The model can serve as a practical screening tool for healthcare providers in resource-limited settings, flagging high-risk patients for confirmatory HbA1c testing.")
    pdf.bullet("Feature importance analysis provides clinically interpretable insights, aligning with established medical knowledge about diabetes risk factors.")

    pdf.sub_title("8.2 Limitations")
    pdf.bullet("The dataset is relatively small (390 patients), which may limit the model's generalizability to other populations.")
    pdf.bullet("The dataset comes from a specific demographic (rural African-American patients), and the model may not perform equally well on other ethnic groups.")
    pdf.bullet("65% of second blood pressure readings were missing, so this potentially useful feature could not be utilized.")
    pdf.bullet("The model was trained on cross-sectional data; longitudinal data could improve predictions.")

    pdf.sub_title("8.3 Future Scope")
    pdf.bullet("Web/Mobile Deployment: Build a simple web application where healthcare workers can input patient data and receive instant diabetes risk predictions.")
    pdf.bullet("Larger Datasets: Train on larger, more diverse datasets (e.g., NHANES, Pima Indians) to improve generalizability.")
    pdf.bullet("Additional Models: Compare with Gradient Boosting (XGBoost), Support Vector Machines, and Neural Networks for potential accuracy improvements.")
    pdf.bullet("Without-Glucose Experiment: Train a separate model excluding glucose to evaluate prediction capability using only non-lab features (demographics + body measurements).")
    pdf.bullet("Explainable AI: Implement SHAP (SHapley Additive exPlanations) values for patient-level prediction explanations.")
    pdf.bullet("Integration with Electronic Health Records (EHR): Integrate the screening tool directly into clinic management software.")

    pdf.ln(6)
    pdf.colored_box(
        "Project Summary",
        "  Problem: Detect diabetes using basic health checkup data\n"
        "  Solution: Random Forest Classifier trained on 18 features from 390 patients\n"
        "  Result: AUC-ROC = 0.956, Accuracy = 92.31%, Recall = 84.62%\n"
        "  Impact: Rapid, accurate diabetes screening for rural healthcare settings",
        (41, 128, 185)
    )

    # ─── Save PDF ─────────────────────────────────────────────
    pdf.output(str(OUTPUT_PDF))
    print(f"\nPDF saved to: {OUTPUT_PDF}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    print("Generating project report PDF...")
    build_report()
    print("Done!")
