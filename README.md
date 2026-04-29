# COVID-19 Patient Symptoms & Diagnosis Dataset
### CSC 405 — Data Science Group Project

---

## Overview

This project analyzes the COVID-19 Patient Symptoms and Diagnosis Dataset to understand the relationship between patient symptoms, demographic characteristics, and COVID-19 diagnosis outcomes. The dataset contains patient-level records including demographic information such as age and gender, along with clinical indicators like fever, body temperature, cough severity, loss of smell, chest pain, and other symptoms. Each record includes a label indicating whether the patient tested positive or negative for COVID-19.

The goal of this project is to perform data cleaning, exploratory data analysis, preprocessing, statistical analysis and hypothesis testing, data modeling, and visualization to identify patterns in symptoms associated with COVID-19 infection. Through EDA, feature analysis, and predictive modeling, the project determines how well patient symptoms and demographic features can predict a COVID-19 diagnosis.

The results can help demonstrate how data science and machine learning support healthcare decision-making, early disease detection, and understanding of symptom patterns in infectious diseases.

---

## Dataset

**File:** `data/covid19_patient_symptoms_diagnosis.csv`
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

```text
COVID-19-Patient-Symptoms-Diagnosis-Dataset/
|
|-- data/
|   |-- covid19_patient_symptoms_diagnosis.csv          // Provided Dataset
|
|-- notebooks/
|   |-- COVID-19-Project-Stage-1.ipynb                  // Saranya Work
|   |-- DataScienceFinalProject.ipynb                   // Saranya Work
|   |-- TitiEDAProject1.ipynb                           // Titilope Work (EDA)
|   |-- ProjectDescription.ipynb                        // Project Overview written by Saranya
|   |-- Project_modeling_mf.ipynb                       // Maisha Work (Modeling)
|   |-- StatisticalAnalysis&VisualizationPart.ipynb     // Saranya Work (Stats & Viz)
|
|-- figures/
|   |-- age_distribution_by_covid_result.png
|   |-- age_outlier_detection.png
|   |-- body_temp_categories_by_covid_result.png
|   |-- body_temp_outlier_detection.png
|   |-- correlation_heatmap.png
|   |-- fever_based_on_gender.png
|   |-- missing_values_heatmap.png
|   |-- oxygen_level_outlier_detection.png
|   |-- patient_count_by_age.png
|   |-- patient_count_by_age_category.png
|   |-- section4_confusionmatrix.png
|   |-- section5_comorbidity_distribution.png
|   |-- section6_outlier_detection.png
|   |-- section6_outlier_detection_fahrenheit.png
|   |-- section8_age_categories.png
|   |-- section9_age_gender_breakdown.png
|   |-- symptom_rate_by_age.png                         
|   |-- covid_positive_by_age.png                       
|
|-- src/
|   |-- FinalSubmissionProject.ipynb     // Final Submission — combined work of all three group members
|
|-- report/
|-- README.md   Formatted by Titilope
```

---

## Analysis Stages

**Data Cleaning & EDA**
- Handle missing values (comorbidity column filled with "No Comorbidities")
- Duplicate detection and removal
- Outlier detection using IQR method on age, oxygen level, and body temperature
- Exploratory analysis of symptom distributions and demographic breakdowns
- Groupby and aggregation analysis by comorbidity, age category, and body temperature category
- Interactive Plotly figures for symptom rate by age group and COVID positive rate by age group

Note: Every memmber of the group did a portion of data cleaning for null and duplicate records on their personal repositories, therefore should be seen as a joint contribution.

**Saranya made contributions to EDA with the creation of the all heatmaps,the age distribution by COVID-19 plot with corresponding code,the body temperature by covid result with corresponding code, and the single panel age, oxygen level and body temperature box plots.**

**Saranya's version of the introductory import set up and introductory descriptive statistics are also included in the Final Submission Project 1 notebook.
Every group member had their own version of import set up and introductory descriptive statistics in their own repositories.**

**All other Exploratory Data Analysis was done by Titilope.**

**Statistical Analysis & Hypothesis Testing**
- Chi-square test used to test the relationship between categorical variables gender and fever
- Null Hypothesis: No relationship between gender and fever
- Alternative Hypothesis: A relationship exists between gender and fever
- Chi-square test used to test the relationship between categorical variables fever and COVID-19 outcome
- Null Hypothesis: No relationship between covid_result and fever
- Alternative Hypothesis: A relationship exists between covid_result and fever
- Independent Two-Sample t-Test used to find relationship between continous numerical variables body_temperature and oxygen_level
- Null Hypothesis: No relationship between body_temperature and oxygen_level
- Alternative Hypothesis: A relationship exists between body_temperature and oxygen_level
- Contingency table analysis with observed and expected values
- Degree of freedom calculated at significance level α = 0.05
- Symptom frequency analysis by COVID result (Positive vs. Negative)

**Note: All Statistical Analysis & Hypothesis Testing present in the Final Project Submission was done by Saranya.**

**Data Modeling**
- Built two machine learning models to predict whether a patient was COVID-19 positive or negative.
- Used age, oxygen level, body temperature, fever, dry cough, sore throat, fatigue, headache, shortness of breath, loss of smell, loss of taste, and chest pain as input features.
- Used covid_result as the target variable (0 = Negative, 1 = Positive).
- Split the dataset into **80% training data** and **20% testing data**.
- Applied **Logistic Regression** as the baseline model.
- Applied **Random Forest** as the advanced model.
- Logistic Regression achieved **81.8% accuracy**.
- Random Forest achieved **86.1% accuracy**.
- Used a confusion matrix to evaluate the Random Forest model.
- Confusion Matrix Results:
  - **340 True Negatives** → correctly predicted negative patients.
  - **136 False Positives** → negative patients predicted as positive.
  - **148 False Negatives** → positive patients predicted as negative.
  - **376 True Positives** → correctly predicted positive patients.

**Note: All Data Modeling present in the Final Project Submission was done by Maisha.**

**Visualization**
- Correlation heatmap across all features
- Missing values heatmap
- Symptom frequency bar charts
- Grouped comparisons by age category, gender, and body temperature category
- Oxygen level group analysis
- Comorbidity distribution before and after cleaning
- Confusion matrix for Random Forest model
- Interactive Plotly charts: symptom rate by age group and COVID positive rate by age group (with mean reference line and color-coded bars)

---

## Key Findings

<<<<<<< HEAD
- Overall COVID-19 positive rate: ~53.8%*
- Most frequent symptoms: fatigue (59%), fever (57%), dry cough (49%)
- Least frequent symptom: loss of taste (29%)
- Patients with oxygen levels in the 85–89 range had the highest COVID-19 positive rate (~67.8%)
- Patients with normal oxygen levels (95–100) had the lowest positive rate (~33.5%)
- Chi-square test revealed a statistically significant relationship between gender and fever at α = 0.05
- Random Forest performed better than Logistic Regression by 4.3%.
- Both models showed strong performance in predicting COVID-19 diagnosis.
- The Random Forest model correctly classified 716 test samples.
- Symptom and clinical features were useful predictors of COVID-19 status.

=======
- Overall COVID-19 positive rate: 52.0%
- Overall COVID-19 negative rate: 48.0%
- Most frequent symptoms: fatigue (59.98%), fever (56.74%), dry cough (49.32%)
- Least frequent symptom: loss of taste (29.28%)
- Patients with oxygen levels in the 85-89 range had the highest COVID-19 positive rate (70.57%)
- Patients with normal oxygen levels (95–100) had the lowest positive rate (~35.94%)
- Chi-square test revealed a statistically significant relationship between COVID-19 and fever at α = 0.05.
  - Chi-square test revealed there is not enough evidence to strongly support a dependent relationship between gender and fever at α = 0.05.
- t-test revealed there is not enough evidence to strongly support that oxygen levels and body temperature affects each other at α = 0.05.
- Random Forest achieved 86.1% accuracy, outperforming Logistic Regression at 81.8%
>>>>>>> 452379263665f665fa4114924c51fcc37db276df

---

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- plotly
- kaleido
- jupyter

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly kaleido jupyter
```

---

## How to Run

1. Clone or download the repository
2. Navigate to the project folder
3. Open the final submission notebook:
```bash
jupyter notebook src/FinalSubmissionProject.ipynb
```
4. Run all cells in order (Kernel → Restart and Run All)

---

## Interactive Figures

The EDA notebook includes interactive Plotly charts that allow you to hover, toggle symptoms, and explore the data visually. To view them fully rendered, open the notebook on nbviewer:

[View FinalSubmissionProject.ipynb on nbviewer](https://nbviewer.org/github/saranya-22-06/COVID-19-Patient-Symptoms-Diagnosis-Dataset/blob/main/src/FinalSubmissionProject.ipynb)

---

## Authors
CSC 405 Data Science, Spring 2026  
Maisha Fyruz, Saranya Yalla, Titilope Adeniyi
