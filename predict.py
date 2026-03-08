"""
Diabetes Detection — Live Prediction Demo
==========================================
Interactive tool that takes patient data and predicts diabetes risk.
Loads the trained Random Forest model and gives instant results.

USAGE:  venv\Scripts\python.exe predict.py

Perfect for live demos during project presentations!
"""

import joblib
import numpy as np
from pathlib import Path
import sys
import os

MODEL_PATH = Path(r"d:\Miniproject\model_outputs\diabetes_rf_model.pkl")

# Feature order must match training
FEATURE_NAMES = [
    'chol', 'stab.glu', 'hdl', 'ratio', 'location', 'age', 'gender',
    'height', 'weight', 'bp.1s', 'bp.1d', 'waist', 'hip', 'time.ppn',
    'BMI', 'waist_hip_ratio', 'frame_large', 'frame_medium', 'frame_small'
]


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    print()
    print("  " + "=" * 58)
    print("  |                                                        |")
    print("  |      DIABETES DETECTION SYSTEM                        |")
    print("  |      Random Forest AI Model                           |")
    print("  |                                                        |")
    print("  |      Enter patient data for instant screening         |")
    print("  |                                                        |")
    print("  " + "=" * 58)
    print()


def get_float(prompt, default=None):
    """Get a float input with optional default."""
    while True:
        try:
            suffix = f" [{default}]" if default is not None else ""
            val = input(f"  {prompt}{suffix}: ").strip()
            if val == "" and default is not None:
                return float(default)
            return float(val)
        except ValueError:
            print("    -> Please enter a valid number.")


def get_choice(prompt, options):
    """Get a choice from a list of options."""
    while True:
        options_str = "/".join(options)
        val = input(f"  {prompt} ({options_str}): ").strip().lower()
        for opt in options:
            if val == opt.lower() or val == opt[0].lower():
                return opt
        print(f"    -> Please enter one of: {options_str}")


def predict_patient():
    """Collect patient data and make a prediction."""

    print("\n  " + "-" * 50)
    print("  PATIENT DATA ENTRY")
    print("  " + "-" * 50)
    print("  (Press Enter to use default/average values)\n")

    # Collect inputs
    age = get_float("Age (years)", 45)
    gender_str = get_choice("Gender", ["Male", "Female"])
    gender = 1 if gender_str.lower() == "male" else 0

    print()
    print("  --- Lab Results ---")
    stab_glu = get_float("Stabilized Glucose (mg/dL)", 100)
    chol = get_float("Total Cholesterol (mg/dL)", 210)
    hdl = get_float("HDL Cholesterol (mg/dL)", 50)
    ratio = chol / hdl if hdl > 0 else 4.5

    print()
    print("  --- Blood Pressure ---")
    bp_1s = get_float("Systolic BP (mmHg)", 130)
    bp_1d = get_float("Diastolic BP (mmHg)", 80)

    print()
    print("  --- Body Measurements ---")
    height = get_float("Height (inches)", 66)
    weight = get_float("Weight (pounds)", 175)
    waist = get_float("Waist (inches)", 37)
    hip = get_float("Hip (inches)", 43)
    frame_str = get_choice("Body Frame", ["Small", "Medium", "Large"])

    print()
    location_str = get_choice("Location", ["Buckingham", "Louisa"])
    location = 0 if location_str.lower() == "buckingham" else 1
    time_ppn = get_float("Time since last meal (minutes)", 300)

    # Engineered features
    bmi = (weight / (height ** 2)) * 703 if height > 0 else 25
    waist_hip = waist / hip if hip > 0 else 0.85

    # Frame encoding
    frame_large = 1 if frame_str.lower() == "large" else 0
    frame_medium = 1 if frame_str.lower() == "medium" else 0
    frame_small = 1 if frame_str.lower() == "small" else 0

    # Build feature vector in the EXACT order the model expects
    features = np.array([[
        chol, stab_glu, hdl, ratio, location, age, gender,
        height, weight, bp_1s, bp_1d, waist, hip, time_ppn,
        bmi, waist_hip, frame_large, frame_medium, frame_small
    ]])

    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    prob_no = probability[0] * 100
    prob_yes = probability[1] * 100

    # Display results
    print("\n")
    print("  " + "=" * 58)

    if prediction == 1:
        print("  |                                                        |")
        print("  |   RESULT:  *** HIGH DIABETES RISK DETECTED ***        |")
        print("  |                                                        |")
        print("  " + "=" * 58)
        print(f"\n  Diabetes Probability: {prob_yes:.1f}%")
        print(f"  Non-Diabetes Probability: {prob_no:.1f}%")
        print("\n  RECOMMENDATION: This patient should undergo a")
        print("  confirmatory HbA1c test immediately.")
    else:
        print("  |                                                        |")
        print("  |   RESULT:  LOW DIABETES RISK                          |")
        print("  |                                                        |")
        print("  " + "=" * 58)
        print(f"\n  Non-Diabetes Probability: {prob_no:.1f}%")
        print(f"  Diabetes Probability: {prob_yes:.1f}%")
        print("\n  RECOMMENDATION: Low risk detected. Continue routine")
        print("  health monitoring and annual checkups.")

    # Show key risk factors
    print("\n  " + "-" * 50)
    print("  RISK FACTOR SUMMARY")
    print("  " + "-" * 50)

    flags = []
    if stab_glu > 126:
        flags.append(f"  [!] Glucose is HIGH ({stab_glu:.0f} mg/dL > 126)")
    elif stab_glu > 100:
        flags.append(f"  [~] Glucose is BORDERLINE ({stab_glu:.0f} mg/dL, normal < 100)")
    else:
        flags.append(f"  [OK] Glucose is NORMAL ({stab_glu:.0f} mg/dL)")

    if bmi > 30:
        flags.append(f"  [!] BMI indicates OBESITY ({bmi:.1f}, normal < 25)")
    elif bmi > 25:
        flags.append(f"  [~] BMI indicates OVERWEIGHT ({bmi:.1f}, normal < 25)")
    else:
        flags.append(f"  [OK] BMI is NORMAL ({bmi:.1f})")

    if bp_1s > 140:
        flags.append(f"  [!] Blood Pressure is HIGH ({bp_1s:.0f}/{bp_1d:.0f} mmHg)")
    elif bp_1s > 120:
        flags.append(f"  [~] Blood Pressure is ELEVATED ({bp_1s:.0f}/{bp_1d:.0f} mmHg)")
    else:
        flags.append(f"  [OK] Blood Pressure is NORMAL ({bp_1s:.0f}/{bp_1d:.0f} mmHg)")

    if age >= 60:
        flags.append(f"  [~] Age is a risk factor ({int(age)} years, higher risk above 45)")
    elif age >= 45:
        flags.append(f"  [~] Age is MODERATE risk ({int(age)} years)")
    else:
        flags.append(f"  [OK] Age is LOW risk ({int(age)} years)")

    if chol > 240:
        flags.append(f"  [!] Cholesterol is HIGH ({chol:.0f} mg/dL > 240)")
    elif chol > 200:
        flags.append(f"  [~] Cholesterol is BORDERLINE ({chol:.0f} mg/dL)")
    else:
        flags.append(f"  [OK] Cholesterol is NORMAL ({chol:.0f} mg/dL)")

    for f in flags:
        print(f)

    print("\n  Legend: [!] = High Risk  |  [~] = Borderline  |  [OK] = Normal")
    print()


def preloaded_demos():
    """Run pre-built demo cases for quick presentations."""
    demos = [
        {
            "name": "Healthy Young Patient",
            "data": [180, 85, 55, 3.3, 0, 28, 0, 65, 145, 118, 72, 32, 38, 720,
                     (145/(65**2))*703, 32/38, 0, 1, 0],
        },
        {
            "name": "High-Risk Elderly Patient",
            "data": [280, 220, 30, 9.3, 1, 68, 1, 67, 230, 165, 95, 48, 45, 120,
                     (230/(67**2))*703, 48/45, 1, 0, 0],
        },
        {
            "name": "Borderline Case",
            "data": [215, 130, 42, 5.1, 0, 52, 0, 63, 195, 142, 88, 40, 46, 300,
                     (195/(63**2))*703, 40/46, 0, 1, 0],
        },
    ]

    for demo in demos:
        features = np.array([demo["data"]])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        status = "DIABETIC" if pred == 1 else "NON-DIABETIC"
        risk_pct = prob[1] * 100

        print(f"\n  Patient: {demo['name']}")
        print(f"  Prediction: {status}  |  Diabetes Probability: {risk_pct:.1f}%")
        print(f"  " + "-" * 50)


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!\n")

    while True:
        clear_screen()
        print_banner()
        print("  Choose an option:")
        print("  [1] Enter patient data manually")
        print("  [2] Run pre-built demo cases (for presentation)")
        print("  [3] Exit")
        print()

        choice = input("  Your choice (1/2/3): ").strip()

        if choice == "1":
            predict_patient()
            input("\n  Press Enter to continue...")
        elif choice == "2":
            preloaded_demos()
            input("\n  Press Enter to continue...")
        elif choice == "3":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice. Try again.")
            input("\n  Press Enter to continue...")
