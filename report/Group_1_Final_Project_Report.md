# Final Project Report — Group 1

## Introduction

The COVID-19 pandemic generated an enormous volume of patient data, creating an urgent need for analytical tools capable of identifying infection patterns and supporting clinical decision-making. This project applies a full data science pipeline — spanning data cleaning, exploratory data analysis, statistical hypothesis testing, and predictive modeling — to a synthetic COVID-19 patient symptoms dataset containing 5,000 records and 18 variables. The central question addressed is: given a patient's demographic profile, reported symptoms, and clinical measurements, can we accurately predict whether that patient is COVID-19 positive or negative?

This report reflects the work of Group 1 and is organized across three analytical sections:
- **Titilope's section** covers data cleaning and exploratory data analysis
- **Saranya's section** covers statistical analysis and hypothesis testing
- **Maisha's section** covers data modeling analysis, comparing Logistic Regression against Random Forest

---

## Exploratory Data Analysis Section

### Data Cleaning Preface

Before any analysis can be done on the COVID-19 dataset, it needs to be cleaned. Raw data often contains issues like missing values, duplicate records, or outliers that can produce inaccurate or misleading results if not corrected. Using the `pandas` and `numpy` libraries, the cleaning process was straightforward and provided practical experience with the necessary steps to prepare a real-world dataset for analysis.

### Dataset Overview

The dataset has 5,000 rows and 18 columns. Each row represents a unique patient record. The dataset includes demographic information, symptom indicators, clinical measurements, and a COVID-19 diagnostic result.

### Data Inspection

The initial inspection reveals that the dataset contains a mix of binary symptom columns (0 or 1), categorical columns (gender, comorbidity), and numerical columns (age, oxygen_level, body_temperature).

#### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `patient_id` | Integer | Unique patient identifier |
| `age` | Integer | Patient age in years |
| `gender` | Categorical | Male or Female |
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
| `covid_result` | Binary (0/1) | Target variable: COVID-19 diagnosis (1 = Positive) |

---

### Demographic Summary

The dataset contains records from patients ranging across multiple age groups with a mean age of around 44 years old. The gender breakdown shows a relatively balanced distribution between male and female patients.

### Handling Duplicate Records

Using `pandas.duplicated()`, the check returned zero duplicate rows, confirming that every row represents a unique patient. No records needed to be removed.

---

### Checking for Missing Values

After confirming no duplicates, missing values were checked across all 18 columns using `.isnull()`. Results showed that only one column contained missing data: **`comorbidity`**.

The `comorbidity` column was missing **2,725 out of 5,000 values** (54.5% of the dataset). All other columns were 100% complete.

#### Filling Missing Values

Rather than deleting the 2,725 rows with missing comorbidity entries, those values were filled with the label `"No Comorbidities"`. This decision was made for two reasons:

1. **Dropping over half of the dataset would make the analysis unreliable.** Removing 2,725 rows would skew the remaining data and make it appear as though all patients had underlying conditions — highly unrealistic.
2. **Missing entries are insights too.** People often do not disclose or do not know whether they have an underlying condition. Labeling those rows "No Comorbidities" preserves the clinical meaning of the data.

#### Comorbidity Distribution Before and After Cleaning

| Comorbidity Group | Count Before Cleaning | Count After Cleaning |
|---|---|---|
| No Comorbidities | 0 (0%) | 2,725 (54.5%) |
| Diabetes | 1,001 (20.0%) | 1,001 (20.0%) |
| Heart Disease | 792 (15.8%) | 792 (15.8%) |
| Asthma | 482 (9.6%) | 482 (9.6%) |

> *Figure 1: The "No Comorbidities" group grows from 0 to 2,725 patients after missing values are filled, while all other groups remain unchanged.*

> *Figure 2: Missing Values per Column — only the `comorbidity` column contains missing data (2,725 entries). All 17 other columns are fully complete.*

---

### Outlier Detection: Age, Oxygen Level, and Body Temperature

#### Methodology: Interquartile Range (IQR)

Any value falling below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR` is marked as a potential outlier.

| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers Found |
|---|---|---|---|---|---|---|
| `age` | 22.00 | 66.00 | 44.00 | -44.00 | 132.00 | 0 |
| `oxygen_level` | 88.00 | 96.00 | 8.00 | 76.00 | 108.00 | 0 |
| `body_temperature` | 37.30 | 39.70 | 2.40 | 33.70 | 43.30 | 0 |

#### Result: Zero Outliers Found

No outliers were detected across any of the three numerical columns. No rows needed to be removed.

> *Figure 2: Outlier Detection — Numerical Columns. Box plots for Age, Oxygen Level (%), and Body Temperature (°C and °F). Red dashed lines mark normal clinical thresholds; orange dashed lines mark dataset means.*

---

### Key Clinical Observations from Descriptive Statistics

#### Oxygen Level: Below the Normal Threshold

The normal threshold for blood oxygen saturation is **95%**. The mean oxygen level across all 5,000 patients is **91.9%**, which falls below the healthy range. This is consistent with a population of patients who were actively experiencing reduced oxygen saturation (hypoxemia), one of the most recognized warning signs of COVID-19.

#### Body Temperature: Above the Normal Threshold

The normal threshold for body temperature is **37°C (98.6°F)**. The mean body temperature across all patients is **38.5°C (101.3°F)**, indicating the average patient was presenting with a fever — one of the most commonly reported symptoms of COVID-19.

> These observations were not flagged by the IQR algorithm because they fall within the statistical range of the dataset. They emerged from understanding the clinical context and comparing dataset averages against established medical thresholds.

---

### Age Distribution Analysis

> *Figure 3: Patient Count by Age. The distribution is relatively even across all age groups with no single age dominating, and no extreme spikes or gaps.*

---

### Age Category Grouping

Patients were grouped into six age categories based on standard demographic life stages:

| Age Category | Age Range |
|---|---|
| Minors | 0–17 |
| Young Adults | 18–34 |
| Adults | 35–49 |
| Middle Aged | 50–59 |
| Senior | 60–79 |
| Elderly | 80+ |

> *Figure 4: Patient Count by Age Category. The Senior group (60–79) has the largest representation at 1,084 patients, while the Elderly (80+) and Middle Aged (50–59) groups are the smallest.*

| Age Category | Count |
|---|---|
| Senior (60–79) | 1,084 |
| Minors (0–17) | 975 |
| Young Adults (18–34) | 976 |
| Adults (35–49) | ~861 |
| Middle Aged (50–59) | 560 |
| Elderly (80+) | 544 |

---

### Age and Gender Breakdown

#### Age Statistics by Gender

| Gender | Mean | Median | Min | Max | Count | Positive Rate (%) |
|---|---|---|---|---|---|---|
| Female | 44.64 | 45.0 | 1 | 89 | 2,514 | 51.47 |
| Male | 44.12 | 44.0 | 1 | 89 | 2,486 | 52.53 |

#### Gender Split Within Each Age Category (%)

| Age Category | Female (%) | Male (%) |
|---|---|---|
| Minors (0–17) | 48.62% | 51.38% |
| Young Adult (18–34) | 50.92% | 49.08% |
| Adult (35–49) | 52.03% | 47.97% |
| Middle Aged (50–59) | 47.68% | 52.32% |
| Senior (60–79) | 51.38% | 48.62% |
| Elderly (80+) | 49.82% | 50.18% |

#### Age Distribution Within Each Gender (%)

| Age Category | Female (%) | Male (%) |
|---|---|---|
| Minors (0–17) | 18.85% | 20.15% |
| Young Adult (18–34) | 19.77% | 19.27% |
| Adult (35–49) | 17.82% | 16.61% |
| Middle Aged (50–59) | 10.62% | 11.79% |
| Senior (60–79) | 22.16% | 21.20% |
| Elderly (80+) | 10.78% | 10.98% |

> *Figure 5: Gender Split Within Each Age Category (%). The male and female split is nearly equal across all age groups — no group skews more than 4% in either direction.*

---

### Overall Symptom Frequency

| Symptom | Overall Count | Overall Rate (%) |
|---|---|---|
| `fatigue` | 2,950 | 58.98 |
| `fever` | 2,850 | 56.74 |
| `dry_cough` | 2,450 | 49.32 |
| `headache` | 2,350 | 44.48 |
| `sore_throat` | 2,250 | 41.60 |
| `shortness_of_breath` | 2,150 | 34.84 |
| `chest_pain` | 2,000 | 30.60 |
| `loss_of_smell` | 1,750 | 29.94 |
| `loss_of_taste` | 1,450 | 29.28 |

> *Figure 7: Symptom Rate — Overall vs COVID Positive vs COVID Negative. Fatigue, fever, and dry cough are the most frequently reported symptoms overall. COVID positive patients consistently show higher rates across all symptoms.*

---

### Symptom Rate by Age Group (%)

| Symptom | Minors (0–17) | Young Adult (18–34) | Adult (35–49) | Middle Aged (50–59) | Senior (60–79) | Elderly (80+) |
|---|---|---|---|---|---|---|
| `fever` | 58.46% | 57.89% | 53.31% | 59.29% | 55.26% | 57.35% |
| `dry_cough` | 48.10% | 49.80% | 47.97% | 50.89% | 50.18% | 49.45% |
| `sore_throat` | 40.72% | 44.36% | 43.09% | 38.93% | 38.93% | 43.93% |
| `fatigue` | 59.08% | 58.61% | 59.70% | 59.11% | 59.13% | 57.90% |
| `headache` | 43.90% | 44.36% | 44.13% | 42.14% | 45.94% | 45.77% |
| `shortness_of_breath` | 36.21% | 35.14% | 33.91% | 32.86% | 33.76% | 37.50% |
| `loss_of_smell` | 30.36% | 29.61% | 29.62% | 32.68% | 29.98% | 27.39% |
| `loss_of_taste` | 31.90% | 27.46% | 28.69% | 30.54% | 27.68% | 30.70% |
| `chest_pain` | 30.67% | 31.86% | 32.17% | 32.86% | 31.51% | 30.88% |

---

### Final Data Analysis Insights

The data exploration and cleaning process confirmed that the dataset is in strong overall condition:

- **No duplicate records** were found
- **No statistical outliers** were detected in any numerical column
- The only data quality concern was the 54.5% missing rate in `comorbidity`, resolved by filling with `"No Comorbidities"`
- Mean oxygen level of **91.9%** is below the 95% healthy threshold
- Mean body temperature of **38.5°C (101.3°F)** is above the 37°C normal threshold

Both observations are consistent with a population of patients actively presenting COVID-19 symptoms.

---

## Statistical Analysis and Hypothesis Testing Section

### Correlation Matrix

A correlation heatmap was generated to examine relationships between variables. Correlation values range from -1 to +1.

#### Key Observations: Correlation with COVID-19 Result

| Feature | Correlation | Interpretation |
|---|---|---|
| `dry_cough` | ~+0.35 | Moderately positive |
| `contact_with_patient` | ~+0.35 | Moderately positive |
| `fever` | ~+0.33 | Moderate positive |
| `loss_of_smell` | ~+0.33 | Moderate positive |
| `shortness_of_breath` | ~+0.32 | Moderate positive |
| `oxygen_level` | ~−0.30 | Moderate negative |

Lower oxygen levels are associated with higher likelihood of infection.

---

### Chi-Square Test 1: Gender vs. Fever

**Objective:** Test the relationship between `gender` and `fever`

- **H₀:** No relationship between gender and fever
- **H₁:** A relationship exists between gender and fever

#### Contingency Table

| Gender | No Fever (0) | Fever (1) | Total |
|---|---|---|---|
| Female | 1,096 | 1,418 | 2,514 |
| Male | 1,067 | 1,419 | 2,486 |
| Total | 2,163 | 2,837 | 5,000 |

*Table 1: Observed values for gender and fever*

#### Results

- χ² = 0.232
- Degrees of freedom = 1
- α = 0.05
- Critical value = 3.841
- p-value = 0.630

**Decision:** Since χ² (0.232) < critical value (3.841) and p = 0.630 > α = 0.05, we **fail to reject H₀**. There is no statistically significant relationship between gender and fever. This is clinically consistent — COVID-19 fever is not systematically more common in one gender.

---

### Chi-Square Test 2: COVID Result vs. Fever

**Objective:** Test the relationship between `covid_result` and `fever`

- **H₀:** No relationship between `covid_result` and fever
- **H₁:** A relationship exists between `covid_result` and fever

#### Contingency Table

| COVID Status | No Fever | Fever | Total |
|---|---|---|---|
| COVID− | 1,444 | 956 | 2,400 |
| COVID+ | 719 | 1,881 | 2,600 |
| Total | 2,163 | 2,837 | 5,000 |

*Table 2: Observed values for `covid_result` and fever*

#### Results

- χ² = 537.46
- Degrees of freedom = 1
- α = 0.05
- Critical value = 3.841
- p-value ≈ 0.000

**Decision:** We **reject H₀**. Fever and COVID-19 result are strongly and significantly associated. COVID-positive patients had a fever rate of **72.35%** compared to only **39.83%** in COVID-negative patients — a difference of **+32.5 percentage points**.

---

### Independent Samples t-Test: Body Temperature vs. Oxygen Level

**Objective:** Compare mean oxygen levels between patients with high vs. normal body temperature

- **H₀:** No significant difference in oxygen levels between the two temperature groups
- **H₁:** A significant difference exists

Patients were split at the median body temperature into High (n ≈ 2,610) and Normal (n ≈ 2,390) groups using `scipy.stats.ttest_ind`.

#### Results

- t = −0.042
- p = 0.967

**Decision:** Since p = 0.967 >> α = 0.05, we **fail to reject H₀**. There is no statistically significant difference in oxygen levels between high and normal body temperature groups. Elevated body temperature and reduced oxygen saturation are independently distributed features in this dataset.

---

### Summary of Hypothesis Testing

| Test | Variables | Statistic | p-value | Decision |
|---|---|---|---|---|
| χ² Test 1 | Gender × Fever | χ² = 0.232 | p = 0.630 | Retain H₀ |
| χ² Test 2 | COVID × Fever | χ² = 537.46 | p ≈ 0.000 | **Reject H₀** |
| t-Test | Temp Groups × O₂ | t = −0.042 | p = 0.967 | Retain H₀ |

*Table 4: Summary of Hypothesis Testing*

---

## Data Modeling Section

After completing data cleaning and EDA, machine learning models were built to predict whether a patient is COVID-19 positive or negative.

### Model Setup

**Target variable:** `covid_result`
- 0 = Negative
- 1 = Positive

**Input features selected:**
- `age`, `oxygen_level`, `body_temperature`
- `fever`, `dry_cough`, `sore_throat`, `fatigue`, `headache`
- `shortness_of_breath`, `loss_of_smell`, `loss_of_taste`, `chest_pain`

### Train-Test Split

The dataset was split using an **80/20 split**:
- Training data: ~4,000 rows
- Testing data: ~1,000 rows

### Models Used

#### Logistic Regression (Baseline)
A simple and commonly used algorithm for binary classification, assuming a linear relationship between input features and the probability of the outcome.

#### Random Forest (Advanced)
Builds multiple decision trees and combines their predictions, capturing more complex relationships between features.

---

### Model Evaluation

**Metric:** Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

| Model | Accuracy |
|---|---|
| Logistic Regression | 81.8% (0.818) |
| Random Forest | **86.1% (0.861)** |

Random Forest outperformed Logistic Regression by approximately **4.3%**.

---

### Confusion Matrix Analysis (Random Forest)

> *Figure: Confusion Matrix for Random Forest model*

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 340 (TN) | 136 (FP) |
| **Actual Positive** | 148 (FN) | 376 (TP) |

- Correct predictions: 340 + 376 = **716**
- Incorrect predictions: 136 + 148 = **284**
- Total test samples: **1,000**

### Interpretation of Results

- **True Positives (376):** Patients correctly identified as COVID positive
- **True Negatives (340):** Patients correctly identified as COVID negative
- **False Positives (136):** Patients incorrectly predicted as positive
- **False Negatives (148):** Patients incorrectly predicted as negative — particularly important in a healthcare context as these represent missed infections

---

### Key Findings

- Random Forest achieved higher accuracy (86.1%) than Logistic Regression (81.8%)
- Both models showed strong performance, indicating the selected features are useful for prediction
- The model correctly classified **716 out of 1,000** test samples (71.6%)
- The presence of false negatives highlights a limitation, especially in a clinical setting
- Symptom and clinical data are effective predictors of COVID-19 diagnosis

---

## Conclusion

This project successfully demonstrated the application of a structured data science pipeline to a COVID-19 clinical dataset across three phases.

**Data Cleaning & EDA:**
- Zero duplicate records and zero statistical outliers across all numerical columns
- Overall COVID-19 positive rate of approximately **52.0%**
- Most frequent symptoms: fatigue (58.98%), fever (56.74%), dry cough (49.32%)
- Least frequent: loss of taste (29.28%)
- Patients with moderate oxygen levels (85–89%) had the highest COVID-19 positive rate at **70.57%**, while patients with normal oxygen levels (95–100%) had the lowest at **35.94%**

**Statistical Analysis & Hypothesis Testing:**
- Gender and fever: **no significant relationship** (p = 0.630)
- COVID-19 diagnosis and fever: **strong significant association** (p ≈ 0.000)
- Body temperature groups and oxygen level: **no significant difference** (p = 0.967)

**Data Modeling:**
- Random Forest outperformed Logistic Regression: **86.1% vs. 81.8%**
- Correctly classified **716 of 1,000** test samples

Taken together, the work completed by Group 1 demonstrates that a well-structured analytical pipeline can extract clinically relevant insights from patient symptom data.
