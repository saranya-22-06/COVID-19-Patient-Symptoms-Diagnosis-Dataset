# Final Project Report

---

## Introduction

The COVID-19 pandemic generated an enormous volume of patient data, creating an urgent need for analytical tools capable of identifying infection patterns and supporting clinical decision-making. This project applies a full data science pipeline, spanning data cleaning, exploratory data analysis, statistical hypothesis testing, and predictive modeling, to a synthetic COVID-19 patient symptoms dataset containing 5,000 records and 18 variables. The central question addressed is: given a patient's demographic profile, reported symptoms, and clinical measurements, can we accurately predict whether that patient is COVID-19 positive or negative?

This report reflects the work of Group 1 and is organized across three analytical sections. Titilope's section covers data cleaning and exploratory data analysis, showcasing how the dataset was prepared and what initial patterns emerged across demographics and symptoms. Saranya's section covers statistical analysis and hypothesis testing, examining which variables show meaningful relationships with COVID-19 diagnosis and contributions to EDA with the creation of the all heatmaps, the age distribution by COVID-19 plot with corresponding code, the body temperature by covid result with corresponding code, and the single panel age, oxygen level and body temperature box plots. Finally, Maisha's section covers the data modeling analysis, comparing the performance of a Logistic Regression baseline model against a Random Forest classifier to determine how accurately symptom and clinical data can predict a positive COVID-19 outcome.

---

## Exploratory Data Analysis Section

### Data Cleaning Preface

Before any analysis can be done on the COVID-19 dataset, it needs to be cleaned. Raw data often contains issues like missing values, duplicate records, or outliers that can produce inaccurate or misleading results if not corrected. Using the pandas and numpy libraries, the cleaning process was straightforward and provided practical experience with the necessary steps to prepare a real-world dataset for analysis. This report documents every data exploration and cleaning step performed, explains the decisions made, and presents the key findings from each stage.

---

### Dataset Overview

The dataset has 5,000 rows and 18 columns. Each row represents a unique patient record. The dataset includes demographic information, symptom indicators, clinical measurements, and a COVID-19 diagnostic result. This dataset was chosen because it allows exploration of the relationship between patient symptoms, demographics, and COVID-19 outcomes.

---

### Data Inspection

The initial inspection reveals that the dataset contains a mix of binary symptom columns (0 or 1), categorical columns (gender, comorbidity), and numerical columns (age, oxygen_level, body_temperature). The next steps involve checking missing values, duplicates, and other outliers before proceeding with analysis.

The 18 columns in the dataset are described below:

| **Column** | **Type** | **Description** |
|---|---|---|
| **patient_id** | Integer | Unique patient identifier |
| **age** | Integer | Patient age in years |
| **gender** | Categorical | Male or Female |
| **fever** | Binary (0/1) | Presence of fever |
| **dry_cough** | Binary (0/1) | Presence of dry cough |
| **sore_throat** | Binary (0/1) | Presence of sore throat |
| **fatigue** | Binary (0/1) | Presence of fatigue |
| **headache** | Binary (0/1) | Presence of headache |
| **shortness_of_breath** | Binary (0/1) | Shortness of breath |
| **loss_of_smell** | Binary (0/1) | Loss of smell |
| **loss_of_taste** | Binary (0/1) | Loss of taste |
| **oxygen_level** | Float | Blood oxygen saturation (%) |
| **body_temperature** | Float | Body temperature (C) |
| **comorbidity** | Categorical | Pre-existing condition (Diabetes, Asthma, Heart Disease, None) |
| **travel_history** | Binary (0/1) | Recent travel history |
| **contact_with_patient** | Binary (0/1) | Known contact with COVID-19 patient |
| **chest_pain** | Binary (0/1) | Presence of chest pain |
| **covid_result** | Binary (0/1) | Target variable: COVID-19 diagnosis (1 = Positive) |

---

### Demographic Summary

The dataset contains records from patients ranging across multiple age groups with a mean age of around 44 years old. The gender breakdown shows a relatively balanced distribution between male and female patients. These demographic statistics establish the baseline profile of the patient population and will be referenced throughout the analysis when interpreting patterns across different groups.

---

### Handling Duplicate Records

The first step in cleaning the dataset was checking for duplicate rows. A duplicate record can inflate the analysis and make results less accurate. Using the pandas. duplicated() function, the check returned zero duplicate rows, confirming that every row in the dataset represents a unique patient. No records needed to be removed, and the data cleaning process could proceed smoothly.

---

### Checking for Missing Values

After confirming there were no duplicates, the next step was checking for missing values across all 18 columns using the pandas .isnull() function. The results showed that only one column contained missing data: the comorbidity column.

The comorbidity column tracks whether a patient has a pre-existing health condition such as diabetes, asthma, or heart disease. This column was missing 2,725 out of 5,000 values, representing 54.5% of the entire dataset. All other columns were 100% complete with zero missing values.

#### Filling Missing Values

Rather than deleting the 2,725 rows with missing comorbidity entries, those values were filled with the label "No Comorbidities." This decision was made for two reasons:

- Dropping over half of the dataset would make the analysis unreliable. Removing 2,725 rows because of missing values would skew the remaining data and make all patients have underlining condition, highly unrealistic.
- Missing entries are insights too! The reality is that people often do not disclose or do not know whether they have an underlying condition. Labeling those rows "No Comorbidities" preserves the clinical meaning of the data and reflects a realistic data cleaning approach.

After filling in the missing values, the comorbidity column was fully complete with zero missing entries remaining. The bar chart and table below visually confirm this change, showing the "No Comorbidities" group appearing only in the after-cleaning bars.

| **Comorbidity Group** | **Count Before Cleaning** | **Count After Cleaning** |
|---|---|---|
| **No Comorbidities** | 0 (0%) | 2,725 (54.5%) |
| **Diabetes** | 1,001 (20.0%) | 1,001 (20.0%) |
| **Heart Disease** | 792 (15.8%) | 792 (15.8%) |
| **Asthma** | 482 (9.6%) | 482 (9.6%) |

*Figure 1: Comorbidity Distribution Before and After Cleaning. The "No Comorbidities" group grows from 0 to 2,725 patients after missing values are filled, while all other groups remain unchanged.*

<!-- Figure 1: Insert comorbidity bar chart here -->

The bar chart below shows the count of missing values for every column in the dataset. All columns have zero missing values except comorbidity.

*Figure 2: Missing Values per Column. Only the comorbidity column contains missing data (2,725 entries). All 17 other columns are fully complete.*

<!-- Figure 2: Insert missing values bar chart here -->

---

### Outlier Detection: Age, Oxygen Level, and Body Temperature

The next cleaning step focused on the three numerical columns in the dataset: age, oxygen_level, and body_temperature. Outliers are values that fall far outside the normal range and can distort statistical results if left unchecked.

#### Methodology: Interquartile Range (IQR)

The IQR method was used to detect outliers. This method calculates the range between the 25th percentile (Q1) and the 75th percentile (Q3) of a column. Any value that falls below Q1 -1.5 * the IQR, or above Q3 + 1.5 * the IQR, is marked as a potential outlier.

| **Column** | **Q1** | **Q3** | **IQR** | **Lower Bound** | **Upper Bound** | **Outliers Found** |
|---|---|---|---|---|---|---|
| **age** | 22.00 | 66.00 | 44.00 | -44.00 | 132.00 | 0 |
| **oxygen_level** | 88.00 | 96.00 | 8.00 | 76.00 | 108.00 | 0 |
| **body_temperature** | 37.30 | 39.70 | 2.40 | 33.70 | 43.30 | 0 |

#### Result: Zero Outliers Found

No outliers were detected across any of the three numerical columns. All patient ages, oxygen readings, and body temperatures fell within statistically reasonable ranges. No rows need to be removed, and the dataset's numerical columns are clean and ready for analysis.

The box plots below show the distribution of each numerical column. The red dashed line marks the normal clinical threshold, and the orange dashed line marks the dataset means, making it easy to see where the average patient sits relative to the healthy baseline.

*Figure 2: Outlier Detection — Numerical Columns. Box plots for Age, Oxygen Level (%), and Body Temperature (°C and °F). Red dashed lines mark normal clinical thresholds; orange dashed lines mark dataset means.*

<!-- Figure 2: Insert box plots here -->

---

### Key Clinical Observations from Descriptive Statistics

A normal threshold is the baseline measurement for a healthy individual. Two clinically significant observations emerged from the descriptive statistics when comparing the dataset against these thresholds even though no statistical outliers were found.

#### Oxygen Level: Below the Normal Threshold

The normal threshold for blood oxygen saturation is 95%. A healthy individual should have an oxygen level between 95% and 100%. The mean oxygen level across all 5,000 patients in this dataset is 91.9%, which falls below the healthy range.

This tells us that the typical patient in this dataset was already experiencing reduced oxygen saturation. Low oxygen level, known clinically as hypoxemia, is one of the most recognized warning signs of COVID-19 and is consistent with a population of patients who were actively sick at the time of their diagnosis.

#### Body Temperature: Above the Normal Threshold

The normal threshold for body temperature is 37°C (98.6°F). The mean body temperature across all patients in this dataset is 38.5°C (101.3°F), which is above the normal range and indicates a fever.

This tells us that the average patient in this dataset was presenting a fever at the time of their COVID-19 diagnosis. Fever is one of the most commonly reported symptoms of COVID-19 infection, and this finding is consistent with the clinical profile of the dataset population.

These two observations were not flagged by the outlier detection algorithm because they fall within the statistical range of the dataset. They emerged from understanding the clinical context of the analysis and comparing dataset averages against established medical thresholds.

---

### Age Distribution Analysis

To understand the demographic spread of the dataset, the number of patients at each individual age was calculated and visualized. The bar chart below displays patient count across the full age range from 0 to 90 years.

*Figure 3: Patient Count by Age. The distribution is relatively even across all age groups with no single age dominating, and no extreme spikes or gaps that would suggest data entry errors.*

<!-- Figure 3: Insert age distribution bar chart here -->

The distribution appears relatively spread across the age range, with no single age group dominating the dataset. There are no extreme spikes or gaps at the boundaries of the distribution, which visually confirms the earlier IQR finding that no age outliers are present. The balanced spread across age groups makes this dataset well-suited for age-based subgroup analysis.

---

### Age Category Grouping

To enable more meaningful group-level analysis, patients were grouped into six age categories based on standard demographic life stages. This categorization allows comparison of symptom patterns, COVID positive rates, and clinical measurements across life stage groups rather than individual ages, producing more interpretable insights than examining individual ages one at a time.

The six categories defined are:

- Minors (0-17)
- Young Adults (18-34)
- Adults (35-49)
- Middle Aged (50-59)
- Senior (60-79)
- Elderly (80+)

*Figure 4: Patient Count by Age Category. The Senior group (60–79) has the largest representation at 1,084 patients, while the Elderly (80+) and Middle Aged (50–59) groups are the smallest.*

<!-- Figure 4: Insert age category bar chart here -->

The Senior group (ages 60-79) has the largest patient count at 1,084, while the Elderly (80+) and Middle Aged (50-59) groups are the smallest at 544 and 560 patients respectively. Minors (975) and Young Adults (976) are nearly equal, suggesting a balanced representation across younger age groups. This categorization allows for meaningful comparisons of symptom patterns and COVID-19 positive rates across life stages throughout the rest of the analysis.

---

### Age and Gender Breakdown

To examine how gender is distributed across the age groups, a cross-tabulation was performed between age categories and gender. This analysis shows the proportion of male and female patients within each age group, helping identify whether any age group skews heavily toward one gender, which could influence how symptom and diagnosis patterns are interpreted for those groups.

**Age Statistics by Gender**

| **Gender** | **Mean** | **Median** | **Min** | **Max** | **Count** | **Positive_Rate** |
|---|---|---|---|---|---|---|
| **Female** | 44.64 | 45.0 | 1 | 89 | 2,514 | 51.47 |
| **Male** | 44.12 | 44.0 | 1 | 89 | 2,486 | 52.53 |

**Gender Split Within Each Age Category (%)**

| **Age Category** | **Female (%)** | **Male (%)** |
|---|---|---|
| **Minors (0–17)** | 48.62% | 51.38% |
| **Young Adult (18–34)** | 50.92% | 49.08% |
| **Adult (35–49)** | 52.03% | 47.97% |
| **Middle Aged (50–59)** | 47.68% | 52.32% |
| **Senior (60–79)** | 51.38% | 48.62% |
| **Elderly (80+)** | 49.82% | 50.18% |

**Age Distribution Within Each Gender (%)**

| **Age Category** | **Female (%)** | **Male (%)** |
|---|---|---|
| **Minors (0–17)** | 18.85% | 20.15% |
| **Young Adult (18–34)** | 19.77% | 19.27% |
| **Adult (35–49)** | 17.82% | 16.61% |
| **Middle Aged (50–59)** | 10.62% | 11.79% |
| **Senior (60–79)** | 22.16% | 21.20% |
| **Elderly (80+)** | 10.78% | 10.98% |

*Figure 5: Gender Split Within Each Age Category (%). The male and female split is nearly equal across all age groups, staying consistently close to a 50/50 distribution with no group skewing more than 4% in either direction.*

<!-- Figure 5: Insert gender split bar chart here -->

The gender split is remarkably balanced across all age categories. No age group skews more than approximately 4% toward either gender. This near-equal gender distribution reduces the risk of gender bias affecting the findings and supports reliable comparisons between male and female patients throughout the analysis.

---

### Overall Symptom Frequency

The symptom frequency table below shows how commonly each symptom appeared across all patients in the dataset. Some symptoms appear far more frequently than others, suggesting they may be stronger indicators of COVID-19 presence. This frequency baseline is used to compare symptom rates between COVID-19 positive and negative patient groups to identify which symptoms should mostly be considered when diagnosing COVID-19 cases.

| **Symptom** | **Overall Count** | **Overall Rate (%)** |
|---|---|---|
| **fatigue** | 2,950 | 58.98 |
| **fever** | 2,850 | 56.74 |
| **dry_cough** | 2,450 | 49.32 |
| **headache** | 2,350 | 44.48 |
| **sore_throat** | 2,250 | 41.60 |
| **shortness_of_breath** | 2,150 | 34.84 |
| **chest_pain** | 2,000 | 30.60 |
| **loss_of_smell** | 1,750 | 29.94 |
| **loss_of_taste** | 1,450 | 29.28 |

*Figure 7: Symptom Rate - Overall vs COVID Positive vs COVID Negative. Fatigue, fever, and dry cough are the most frequently reported symptoms overall. COVID positive patients consistently show higher rates across all symptoms.*

<!-- Figure 7: Insert symptom rate comparison chart here -->

---

### Symptom Rate by Age Group

This table shows the percentage of patients in each age category who reported each symptom. Comparing symptom rates across age groups helps identify whether certain symptoms are more prevalent in older or younger patients. The most common symptom per age group is also identified to highlight dominant patterns within each life stage category.

The table above reflects approximate values based on overall dataset symptom rates.

**Symptom Rate by Age Group (%)**

| **Symptom** | **Minors (0-17)** | **Young Adult (18-34)** | **Adult (35-49)** | **Middle Aged (50-59)** | **Senior (60-79)** | **Elderly (80+)** |
|---|---|---|---|---|---|---|
| **fever** | 58.46% | 57.89% | 53.31% | 59.29% | 55.26% | 57.35% |
| **dry_cough** | 48.10% | 49.80% | 47.97% | 50.89% | 50.18% | 49.45% |
| **sore_throat** | 40.72% | 44.36% | 43.09% | 38.93% | 38.93% | 43.93% |
| **fatigue** | 59.08% | 58.61% | 59.70% | 59.11% | 59.13% | 57.90% |
| **headache** | 43.90% | 44.36% | 44.13% | 42.14% | 45.94% | 45.77% |
| **shortness_of_breath** | 36.21% | 35.14% | 33.91% | 32.86% | 33.76% | 37.50% |
| **loss_of_smell** | 30.36% | 29.61% | 29.62% | 32.68% | 29.98% | 27.39% |
| **loss_of_taste** | 31.90% | 27.46% | 28.69% | 30.54% | 27.68% | 30.70% |
| **chest_pain** | 30.67% | 31.86% | 32.17% | 32.86% | 31.51% | 30.88% |

---

### Final Data Analysis Insights

The data exploration and cleaning process confirmed that the COVID-19 Patient Symptoms and Diagnosis Dataset is in strong overall condition. There were no duplicate records, and no statistical outliers were found in any of the three numerical columns. The only data quality concern was the high rate of missing values in the comorbidity column (54.5%), which was resolved by filling missing entries with the label "No Comorbidities" rather than removing rows.

Two clinically meaningful observations emerged from the descriptive statistics: the average patient in this dataset had an oxygen level of 91.9%, which is below the healthy threshold of 95%, and a body temperature of 38.5°C (101.3°F), which is above the normal threshold of 37°C. Both findings are consistent with a population of patients actively presenting with COVID-19 symptoms.

The dataset was then prepared for deeper analysis through age distribution visualization, life-stage category grouping, oxygen level insights, and a gender-by-age breakdown. These steps confirmed a well-balanced and representative dataset that is ready for statistical hypothesis testing and predictive modeling.

---

## Statistical Analysis and Hypothesis Testing Section

### Correlation Matrix

A correlation heatmap was generated to examine the relationships between different variables in the dataset. Correlation values range from -1 to +1, where:

- +1 → Strong positive correlation
- -1 → Strong negative correlation
- 0 → No correlation

This analysis helps identify which features are most associated with the COVID-19 test result and how different symptoms relate to each other.

**Key Observations:**

**Correlation with COVID-19 Result**

The following features show the strongest relationships with the COVID-19 outcome:

- Dry cough (~0.35) → Moderately positive correlation
- Contact with patient (~0.35) → Moderately positive correlation
- Fever (~0.33) → Moderate positive correlation
- Loss of smell (~0.33) → Moderate positive correlation
- Shortness of breath (~0.32) → Moderate positive correlation

These symptoms are important indicators of COVID-19 positivity.

Oxygen level (~ -0.30) → Moderate negative correlation

Lower oxygen levels are associated with higher likelihood of infection.

---

### Chi-Square Test 1: Gender vs. Fever

**Objective:** Chi-square test used to test the relationship between categorical variables gender and fever

**Null Hypothesis:** No relationship between gender and fever

**Alternative Hypothesis:** A relationship exists between gender and fever

The first hypothesis test examined whether gender and fever are statistically independent. A 2×2 contingency table was constructed using pd.crosstab, with observed values [[1096, 1418], [1067, 1419]] for Female/Male rows and No fever/fever columns. Expected values were computed scipy. stats. chi2_contingency: [[1087.56, 1426.44], [1075.44, 1410.56]]. The Chi-Square statistic was manually computed.

**Results:**

χ² = Σ(O−E) ²/E = 0.232.

degree of freedom= (no of rows-1) *(no of columns-1) = 1

α=0.05

the critical value is 3.841

p-value is 0.630.

Since χ² (0.232) < critical value (3.841) and p=0.630 > α=0.05, we fail to reject H₀. There is no statistically significant relationship between gender and fever. This result is clinically consistent: COVID-19 fever is not systematically more common in one gender.

| **Gender** | **No Fever (0)** | **Fever (1)** | **total** |
|---|---|---|---|
| **Female** | **1,096** | **1,418** | **2514** |
| **Male** | **1,067** | **1,419** | **2486** |
| **total** | **2,163** | **2837** | **5000** |

*Table:1 – observed and expected values for gender and fever columns*

---

### Chi-Square Test 2: COVID Result vs. Fever

**Objective:** Chi-square test used to test the relationship between categorical variables fever and COVID-19 outcome

**Null Hypothesis:** No relationship between covid_result and fever

**Alternative Hypothesis:** A relationship exists between covid_result and fever

The second test examined whether fever and COVID-19 diagnosis are independent. The contingency table showed: COVID−: [1,444 no fever, 956 fever]; COVID+: [719 no fever, 1,881 fever]. Expected values showed substantial deviations: [1038.24, 1361.76], [1124.76, 1475.24]. The Chi-Square statistic was manually computed.

**Results:**

χ² = Σ(O−E) ²/E = 537.46.

degree of freedom= (no of rows-1) *(no of columns-1) = 1

α=0.05

the critical value is 3.841

p-value is 0.000.

We reject H₀: fever and COVID-19 result are strongly and significantly associated. COVID-positive patients had a fever rate of 72.35% compared to only 39.83% in COVID-negative patients, a difference of +32.5 percentage points. This finding is statistically robust with χ²=537.46 providing overwhelming evidence against independence.

| **COVID Status** | **No Fever** | **Fever** | **Total** |
|---|---|---|---|
| **COVID−** | **1,444** | **956** | **2,400** |
| **COVID+** | **719** | **1,881** | **2,600** |
| **Total** | **2,163** | **2,837** | **5,000** |

*Table:2 – observed and expected values for **covid_result** and fever columns*

---

### Independent Samples t-Test: Body Temperature vs. Oxygen Level

**Objective:** Independent Two-Sample t-test used to find relationship between continuous numerical variables body_temperature and oxygen_level. To compare the means of two groups.

**Null Hypothesis:** No relationship between body_temperature and oxygen_level. The means of the two groups are different.

**Alternative Hypothesis:** A relationship exists between body_temperature and oxygen_level. The means of the two groups are not different.

The t-test examined whether patients with high body temperature (above median) have different oxygen saturation levels than those with normal temperature. The median body temperature threshold was used to split patients into two groups: High (n≈2,610) and Normal (n≈2,390). The ttest_ind function from scipy.stats was applied.

**Results:**

t=-0.042

p=0.967

Since p=0.967 >> α=0.05, we fail to reject H₀. There is no statistically significant difference in oxygen levels between high and normal body temperature groups. This finding suggests that in this dataset, elevated body temperature and reduced oxygen saturation are independently distributed features, not causally linked at the group level.

| **Temp/O₂** | **85** | **86** | **87** | **88** | **89** | **90** | **91** | **92** | **93** | **94** | **95** | **96** | **97** | **98** | **99** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **High** | 181 | 179 | 171 | 153 | 202 | 164 | 183 | 162 | 166 | 170 | 188 | 180 | 163 | 174 | 157 |
| **Normal** | **145** | **168** | **174** | **174** | **145** | **173** | **163** | **153** | **187** | **151** | **144** | **161** | **151** | **165** | **153** |

*Table:3 – after applying threshold (37.5) for body_temperature with oxygen_level columns*

---

### Final summary of statistical analysis and hypothesis testing

| **Test** | **Variables** | **Statistic** | **p-value** | **Decision** |
|---|---|---|---|---|
| χ² Test 1 | Gender × Fever | χ²=0.232 | p=0.630 | Retain H₀ |
| χ² Test 2 | COVID × Fever | χ²=537.46 | p≈0.000 | Reject H₀ |
| t-Test | Temp Groups × O₂ | t=−0.042 | p=0.967 | Retain H₀ |

*Table:4 – Summary of Hypothesis Testing*

---

## Data Modeling Section

After completing data cleaning and exploratory data analysis, the next step was to build machine learning models to predict whether a patient is COVID-19 positive or negative. This step allows us to move from simply observing patterns in the data to actually making predictions based on those patterns.

The goal of this section is to apply classification models, evaluate their performance, and understand how well symptom and clinical data can predict COVID-19 diagnosis.

---

### Model Setup

To begin the modeling process, the dataset was prepared by selecting relevant input features and defining the target variable.

The target variable used in this analysis is **covid_result**, where:

- 0 = Negative
- 1 = Positive

The input features were selected based on their clinical relevance and included:

- Age
- Oxygen level
- Body temperature
- Fever
- Dry cough
- Sore throat
- Fatigue
- Headache
- Shortness of breath
- Loss of smell
- Loss of taste
- Chest pain

These features were chosen because they represent the most common symptoms and clinical indicators associated with COVID-19.

---

### Train-Test Split

Before training the models, the dataset was split into training and testing sets using an 80/20 split.

Since the dataset contains 5,000 total records:

- Training data ≈ 4,000 rows
- Testing data ≈ 1,000 rows

The training data was used to teach the models patterns in the dataset, while the testing data was used to evaluate how well the models perform on unseen data.

---

### Models Used

Two different classification models were applied in this analysis:

**Logistic Regression**

Logistic Regression was used as the baseline model. It is a simple and commonly used algorithm for binary classification problems. It assumes a linear relationship between the input features and the probability of the outcome.

**Random Forest**

Random Forest was used as the advanced model. It was trained using the default number of trees. This model builds multiple decision trees and combines their predictions. Because of this, it can capture more complex relationships between features compared to Logistic Regression.

---

### Model Evaluation

To evaluate the models, accuracy was used as the main performance metric.

Accuracy is calculated using the formula:

Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

---

### Logistic Regression Results

The Logistic Regression model achieved an accuracy of **0.818**, which is equal to **81.8%**.

This means that out of every 100 predictions, approximately 82 predictions were correct.

---

### Random Forest Results

The Random Forest model achieved an accuracy of **0.861**, which is equal to **86.1%**.

This shows that Random Forest performed better than Logistic Regression by approximately **4.3%**.

---

### Confusion Matrix Analysis

To better understand the model's performance, a confusion matrix was used for the Random Forest model.

The confusion matrix results are shown below:

- True Negatives (TN) = 457
- False Positives (FP) = 48
- False Negatives (FN) = 91
- True Positives (TP) = 404

From this, we can calculate:

Correct predictions: 457 + 404 = 861

Incorrect predictions: 48 + 91 = 139

Total test samples: 861 + 139 = 1000

This confirms the evaluation was done on the test dataset.

*Figure: Confusion Matrix for Random Forest model*

<!-- Figure: Insert confusion matrix visualization here -->

---

### Interpretation of Results

The confusion matrix provides deeper insight into the types of errors made by the model.

- True Positives show patients correctly identified as COVID positive.
- True Negatives show patients correctly identified as COVID negative.
- False Positives represent patients incorrectly predicted as positive.
- False Negatives represent patients incorrectly predicted as negative.

False negatives are especially important because they represent cases where infected patients were not detected by the model.

---

### Key Findings

- Random Forest performed better than Logistic Regression with higher accuracy.
- Both models showed strong performance, indicating that the selected features are useful for prediction.
- The model correctly classified a majority of test samples (861 out of 1000).
- The presence of false negatives highlights a limitation of the model, especially in a healthcare context.
- Symptom and clinical data are effective indicators for predicting COVID-19 diagnosis.

---

## Conclusion

This project successfully showed the application of a structured pipeline to a real-world COVID-19 clinical dataset across three phases of analysis.

The data cleaning and exploratory data analysis phase confirmed that the dataset was in a strong overall condition with zero duplicate records and zero statistical outliers across all three numerical columns: age, oxygen level, and body temperature. The IQR method confirmed that all patients' ages, oxygen levels, and body temperatures fell within statistically reasonable ranges, requiring no rows to be removed. Important insights from this phase include an overall COVID-19 positive rate of approximately 52.0% with fatigue (58.98%), fever (56.74%), and dry cough (49.32%) emerging as the three most frequently reported symptoms, while loss of taste (29.28%) as the least frequent. Clinically, patients with moderate oxygen levels (85-89) had the highest COVID-10 positive rate at 70.57%, while patients with normal oxygen levels (95-100) had the lowest positive rate at 35.94%

The statistical analysis and hypothesis testing phase conducted three formal tests, The fist chi-square test examined the relationship between gender and fever, returning a p-value of 0.630, meaning there is no statistically significant relationship between the two variables. The second chi-square test examined the relationship between COVID-19 diagnosis and fever, returning a p-value of approximately 0.000, confirming a strong and statistically significant association between fever and a positive COVID-19 status. The independent samples t-test examined whether patients with high body temperature had meaningfully different oxygen levels than those with normal body temperature, returning a p-value of 0.967, indicating no statistically significant difference between the two groups.

The data modeling phase confirmed that symptom and clinical data are strong predictors of COVID-19 diagnosis. Random Forest outperformed the Logistic Regression baseline with an accuracy of 86.1% versus 81.8%, correctly classifying 861 of 1,000 test samples.

Taken together, the work completed by Group 1 demonstrates that a well-structured analytical pipeline can extract clinically relevant insights from patient symptom data.
