# SBA Loan Default Prediction --- California Case Study

## 1. Project Overview

This project examines a credit-risk decision problem from the
perspective of a loan officer at Bank of America in California. The
objective is to estimate the Probability of Default (PD) for two small
businesses applying for SBA-backed loans and assess whether their
applications should be approved or denied.

The case study focuses on **California and NAICS sector 53 (Real Estate
and Rental and Leasing)**.

### The two loan applications

  Business              Loan Requested   SBA Guaranteed Real Estate
  ------------------- ---------------- ---------------- -------------
  Carmichael Realty        \$1,000,000        \$750,000 Yes
  SV Consulting              \$100,000         \$40,000 No

Both applications are considered outside a recession period.

## 2. Dataset

The project uses the U.S. Small Business Administration (SBA) loan
dataset.

-   **National dataset:** `SBAnational.csv`, containing approximately
    899,164 loans from 1987--2014.
-   **Case-study dataset:** `SBAcase.csv`, a California subset
    restricted to NAICS sector 53.
-   **Target variable:** `Default`, where 1 represents a charged-off
    loan and 0 represents a loan paid in full.

The SBA guarantee covers part of a lender's loss if a borrower defaults.
The lender remains exposed to the non-guaranteed portion, making
credit-risk assessment important.

## 3. Why I Changed the Original Methodology

I initially followed the approach presented in the *Journal of
Statistics Education* article:

Li, Mickel & Taylor (2018), *Should This Loan be Approved or Denied?*

I began with the three explanatory variables used in the original
California case study:

-   `RealEstate`
-   `Portion`
-   `Recession`

I then tested a five-variable specification by adding:

-   `New`
-   `DisbursementGross`

The results were disappointing: the three-variable and five-variable
specifications generally produced ROC-AUC values in the 0.70--0.80
range. These results did not provide the improvement I was looking for.
I did not want to stop at reproducing the original analysis, so I
decided to investigate additional credit-scoring techniques and broaden
the information available to the models.

In particular, I introduced **Weight of Evidence (WoE)** and
**Information Value (IV)** to transform and screen additional
explanatory variables. The purpose was to move beyond the original
limited-variable specification while maintaining a credit-scoring
approach.

### References and inspiration

-   **[Journal of Statistics Education --- Should This Loan be Approved
    or Denied?](https://doi.org/10.1080/10691898.2018.1434342):** The
    original California-based case study. It provided the initial
    problem framing, the original explanatory variables and the
    classification cutoff retained in this project.
-   **[credit-scorecard-development](https://github.com/wavetnsch/credit-scorecard-development):**
    A practical GitHub reference that inspired parts of my WoE/IV and
    credit-scorecard implementation. I used it as a methodological
    reference and adapted ideas and implementation elements to the SBA
    dataset and the requirements of this case study.

The evaluation framework was also expanded. ROC-AUC alone cannot
establish whether a credit-risk model is reliable: a model may rank
borrowers effectively but still produce poorly calibrated probabilities,
overfit its training data or become unstable across economic periods and
borrower segments.

## 4. Estimation Methodology

The final approach retains the five original case-study variables and
adds further explanatory variables through WoE transformation and
IV-based feature selection.

### 4.1 Original variables

  -----------------------------------------------------------------------
  Variable                            Description
  ----------------------------------- -----------------------------------
  `Recession`                         Indicates whether the loan period
                                      overlaps with a recession

  `New`                               Indicates whether the business is
                                      two years old or less

  `RealEstate`                        Real-estate-backed loan indicator

  `DisbursementGross`                 Gross amount disbursed

  `Portion`                           Percentage of the loan guaranteed
                                      by the SBA
  -----------------------------------------------------------------------

Additional available borrower and loan characteristics are considered as
candidate features and screened using WoE/IV.

### 4.2 Weight of Evidence (WoE)

WoE transforms numerical or categorical variables into values
representing the relative distribution of defaults and non-defaults
within each group.

The implementation includes:

-   Automatic numerical binning.
-   WoE calculation.
-   Laplace smoothing to handle zero-frequency bins.
-   Information Value calculation.
-   Automatic feature screening.
-   Investigation of unusually high IV values for potential leakage.

The sign convention used must be interpreted consistently with the
implementation. In this project, WoE is defined as:

$$
\mathrm{WoE}_i = \ln\left(
\frac{\text{Distribution of Events}_i}
{\text{Distribution of Non-Events}_i}
\right)
$$

where:

- **Events** are loans that defaulted (`Default = 1`).
- **Non-events** are loans that were paid in full (`Default = 0`).

The event distribution for bin \(i\) is the number of defaults in that
bin divided by the total number of defaults. The non-event distribution
is calculated in the same way using non-defaulted loans.

With this convention, a **positive WoE** indicates that the bin has a
relatively higher concentration of defaults, while a **negative WoE**
indicates a relatively higher concentration of non-defaults. This sign
convention is maintained consistently throughout WoE transformation
and model interpretation.

### 4.3 Information Value (IV)

IV measures how well a variable separates defaulted and non-defaulted
loans.

Variables with IV below 0.02 are treated as weak candidates under the
adopted screening rule. Variables with IV above 0.50 are investigated
carefully because unusually high predictive power may indicate leakage
or a suspicious relationship. Variables containing post-default
information, such as charge-off or recovery details, must be excluded
from a model intended to estimate risk at application time.

### 4.4 Models

Only two estimation approaches are retained:

1.  **WoE + Logistic Regression:** An interpretable credit-scoring
    approach for estimating PD.
2.  **WoE + XGBoost:** A gradient-boosting approach that can capture
    more complex relationships.

MinMax scaling is applied only where appropriate. Tree-based XGBoost
does not generally require MinMax normalization.

### 4.5 Hyperparameter Optimization

The models are optimized using randomized hyperparameter search. The
search considers relevant WoE/IV settings, Logistic Regression
regularization and XGBoost parameters such as tree depth, learning rate,
number of estimators, subsampling and regularization.

Hyperparameter selection must be performed using training data only. The
held-out test set must not be used to select features, tune parameters
or repeatedly adjust the model.

## 5. Model Performance

The initial experiments motivated the methodological changes described
above.

### Earlier model comparison

  Model                                 ROC-AUC
  ----------------------------------- ---------
  Logistic Regression (3 variables)       0.754
  Logistic Regression (5 variables)       0.740
  XGBoost (5 variables)                   0.805
  SVM (5 variables)                       0.760
  Logistic Regression + WoE               0.798
  XGBoost + WoE                           0.840

### Improved model performance

The latest reported held-out test results are:

  Model                           ROC-AUC
  --------------------------- -----------
  WoE + Logistic Regression     **0.931**
  WoE + XGBoost                 **0.950**

These results represent a substantial improvement over the earlier
experiments. XGBoost achieved the highest reported ROC-AUC, while WoE
Logistic Regression also showed strong discriminatory performance.

These are reported test-set results, not proof of production readiness.
They must be interpreted alongside cross-validation, temporal
validation, calibration and stability results.

## 6. Evaluation Metrics

The evaluation framework goes beyond ROC-AUC.

  -----------------------------------------------------------------------
  Metric                              Purpose
  ----------------------------------- -----------------------------------
  ROC-AUC                             Measures the ability to rank
                                      defaulters above non-defaulters

  Gini coefficient                    Discriminatory power, calculated as
                                      Gini = (2 × ROC-AUC) − 1

  KS statistic                        Maximum separation between default
                                      and non-default score distributions

  Confusion matrix                    Shows true positives, false
                                      positives, true negatives and false
                                      negatives

  Precision                           Proportion of predicted defaults
                                      that actually defaulted

  Recall                              Proportion of actual defaults
                                      identified

  F1-score                            Harmonic mean of precision and
                                      recall

  Brier score                         Measures the accuracy of predicted
                                      probabilities

  Log Loss                            Penalizes inaccurate probability
                                      predictions
  -----------------------------------------------------------------------

## 7. Model Validation and Reliability

A strong test-set AUC does not automatically establish that a model is
reliable. The following validation procedures are used to investigate
generalization, temporal performance, calibration and stability.

### 7.1 Overfitting check

Compare training ROC-AUC with test ROC-AUC. A gap greater than 0.05 is
flagged for investigation.

### 7.2 Dummy classifier baseline

Compare the models against a `DummyClassifier` using the majority-class
strategy. This provides a basic benchmark to check that the models offer
predictive value beyond always predicting the most frequent class.

### 7.3 Stratified 5-fold cross-validation

Use five folds while preserving the default/non-default proportions in
each fold. Report the mean ROC-AUC and standard deviation to assess
performance consistency.

### 7.4 Expanding-window temporal cross-validation

Train on the oldest available loans and evaluate on subsequent newer
observations. Progressively expand the training window to simulate model
use over time.

### 7.5 Rolling-window temporal cross-validation

Use a fixed-size historical training window that moves forward through
time. This tests whether the model remains effective as older
observations are replaced by newer ones.

### 7.6 Probability calibration

Use calibration curves and Brier scores to assess whether predicted PDs
correspond to observed default frequencies. A model should not only rank
borrowers effectively; its estimated probabilities should also be
meaningful.

### 7.7 Population Stability Index (PSI)

Use PSI to compare distributions between reference and comparison
populations and identify potential population shifts.

  PSI          Practical interpretation
  ------------ ------------------------------
  Below 0.10   Relatively stable
  0.10--0.25   Moderate population shift
  Above 0.25   Significant population shift

These are practical guidelines, not universal regulatory thresholds.

### 7.8 Stability across borrower segments

Compare model performance across:

-   Real-estate-backed and non-real-estate-backed loans.
-   Recession and non-recession periods.
-   Different loan-size categories.

Segment-level results should be interpreted cautiously when a segment
contains few observations or very few defaults.

## 8. Credit Score Scaling --- PDO Method

For WoE Logistic Regression, score scaling can be implemented using the
Points to Double the Odds (PDO) approach:

\[ Score = Offset + Factor `\times `{=tex}`\ln`{=tex}(Odds) \]

The reference configuration is:

-   Base score = 600.
-   Base odds = 50:1 (non-default to default).
-   PDO = 20.

The score convention should be checked against the model's predicted
odds so that higher scores consistently correspond to lower default
risk.

## 9. Scenario-Based Credit Decisions

The final stage applies the two models to the original California loan
applications:

-   Carmichael Realty.
-   SV Consulting.

For each business, the models estimate PD and produce a classification
using the selected cutoff.

The classification cutoff of **0.50** is retained from the original
*Journal of Statistics Education* California case study to preserve
consistency with the assignment. It is not presented as an optimized
banking threshold.

## 10. Limitations

-   The dataset contains approved SBA loans, which can introduce
    selection bias.
-   The study is restricted to California and NAICS sector 53.
-   The observations are historical and may not represent current
    economic conditions.
-   Some borrower characteristics are unavailable or incomplete.
-   Segment-level metrics may be uncertain when sample sizes are small.
-   A high ROC-AUC does not by itself establish reliable PD calibration
    or temporal generalization.

The model should be interpreted as an analytical credit-risk
decision-support exercise, not a complete representation of a bank's
underwriting process.

## 11. Final Objective

This project began as an attempt to reproduce and extend an existing
statistical case study. The initial three-variable and five-variable
models did not deliver the performance I was seeking. Rather than
accepting those results, I expanded the feature-engineering process
using WoE and IV, drew inspiration from practical credit-scorecard
implementations, optimized the two selected models and introduced a
broader evaluation and validation framework.

The objective is not simply to obtain a high AUC. It is to build a model
that can distinguish risky from non-risky loans, estimate meaningful
default probabilities, remain stable across borrower populations and
economic periods, and support more explainable credit decisions.
