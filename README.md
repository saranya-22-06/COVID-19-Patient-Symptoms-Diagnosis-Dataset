# COVID-19 Patient Symptoms & Diagnosis Dataset
### CSC 405 — Data Science Group Project

---

## Overview

This project analyzes a COVID-19 patient symptoms dataset to identify patterns between patient symptoms, demographic characteristics, and COVID-19 diagnosis outcomes. The analysis applies data science techniques including exploratory data analysis (EDA), statistical hypothesis testing, and predictive modeling to determine how well patient symptoms can predict a COVID-19 positive diagnosis.

---

## Dataset

**File:** `Desktop/405 GROUP PROJECT/data/covid19_patient_symptoms_diagnosis.csv`  
**Records:** 5,000 patients  
**Features:** 18 columns

| Column | Type | Description |
|---|---|---|
| `patient_id` | Integer | Unique patient identifier |
| `age` | Integer | Patient age in years |
| `gender` | Categorical | Male / Female |
| `fever` | Binary (0/1) | Presence of fever |
| `dry_cough` | Binary (0/1) | Presence of dry cough |
| `sore_throat` | Binary (0/1) | Presence of sore throat |
| `fatigue` | Binary (0/1) | Presence of fatigue |
| `headache` | Binary (0/1) | Presence of headache |
| `shortness_of_breath` | Binary (0/1) | Shortness of breath |
| `loss_of_smell` | Binary (0/1) | Loss of smell |
| `loss_of_taste` | Binary (0/1) | Loss of taste |
| `oxygen_level` | Float | Blood oxygen saturation (%) |
| `body_temperature` | Float | Body temperature (°C) |
| `comorbidity` | Categorical | Pre-existing condition (Diabetes, Asthma, Heart Disease, None) |
| `travel_history` | Binary (0/1) | Recent travel history |
| `contact_with_patient` | Binary (0/1) | Known contact with COVID-19 patient |
| `chest_pain` | Binary (0/1) | Presence of chest pain |
| `covid_result` | Binary (0/1) | **Target variable** — COVID-19 diagnosis (1 = Positive) |

---

## Project Structure
COVID-19-Patient-Symptoms-Diagnosis-Dataset-main/
│
├── Desktop/405 GROUP PROJECT/
│   ├── Project1.ipynb                  # Main group project notebook
│   ├── Project1 Original.ipynb         # Original version
│   └── data/
│       └── covid19_patient_symptoms_diagnosis.csv
│
├── COVID-19-Project-Stage-1.ipynb      # Stage 1: Data cleaning & EDA
├── StatisticalAnalysis&VisualizationPart.ipynb  # Statistical analysis
├── Project_modeling.ipynb              # Predictive modeling
├── FinalSubmissionProject.ipynb        # Final combined submission
├── DataScienceFinalProject.ipynb       # Full project notebook
└── ProjectDescription.ipynb           # Project description and objectives
---

## Analysis Stages

**Stage 1 — Data Cleaning & EDA**
- Handle missing values (comorbidity column had ~54.5% missing → filled with "No Comorbidities")
- Outlier detection using IQR method on age, oxygen level, and body temperature
- Exploratory analysis of symptom distributions and demographic breakdowns

**Stage 2 — Statistical Analysis & Hypothesis Testing**
- Chi-square tests for categorical symptom variables vs. COVID-19 outcome
- T-tests for continuous variables (oxygen level, body temperature) vs. outcome
- Symptom frequency analysis by COVID result (Positive vs. Negative)

**Stage 3 — Data Modeling**
- Feature engineering and preprocessing
- Predictive modeling to classify COVID-19 diagnosis outcome
- Model evaluation using accuracy, precision, recall, and F1 score

**Stage 4 — Visualization**
- Symptom frequency bar charts
- Grouped comparisons (Overall vs. COVID+ vs. COVID-)
- Oxygen level group analysis
- Comorbidity distribution plots

---

## Key Findings

- Overall COVID-19 positive rate: **~53.8%**
- Most frequent symptoms: fatigue (59%), fever (57%), dry cough (49%)
- Least frequent symptom: loss of taste (29%)
- Patients with oxygen levels in the 85–89 range had the highest COVID-19 positive rate (~67.8%)
- Patients with normal oxygen levels (95–100) had the lowest positive rate (~33.5%)

---

## Requirements
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## How to Run

1. Clone or download the repository
2. Navigate to the project folder
3. Open the desired notebook in Jupyter:
```bash
jupyter notebook FinalSubmissionProject.ipynb
```
4. Run all cells in order (Kernel → Restart & Run All)

---

## Authors

Titilope, Maisha, Saranya Group 1 CSC 405 Data Science, Spring 2026
