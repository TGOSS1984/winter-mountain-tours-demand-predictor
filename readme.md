<div align="center">

# Winter Mountain Tour Demand & Cancellation Predictor

[![Heroku Deployment](https://img.shields.io/badge/Heroku-Live%20App-79589F?style=for-the-badge&logo=heroku&logoColor=white)](https://winter-tour-predictor-ce48d589f61d.herokuapp.com/)
![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF9900?style=for-the-badge&logo=xgboost&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

<div align="center">
<img src="app_pages/assets/images/readme_logo.png" alt="Logo" width="80" height="80">
</div>

---

A Streamlit-based predictive analytics app that forecasts **weekly winter tour bookings by region** and predicts **cancellation risk** for individual bookings.

This project is built for **Portfolio Project 5: Predictive Analytics** on the Code Institute **Full Stack Software Development Diploma (Predictive Analytics)**.

It extends the domain established in **Portfolio Project 4: Winter Mountain Tours V2**, where I built a full-stack Django booking system for UK winter mountain routes.  
PP4 focused on the **operational layer** (booking, managing, and cancelling tours).  
View PP4 here: [Live Site](https://uk-winter-mountain-tours-v2-c6f21d80d2c8.herokuapp.com/)

PP5 adds the **decision-support analytics layer**: forecasting demand, predicting cancellations, and surfacing insights about weather, holidays, and seasonality.

This project has been a major learning curve for me, particularly in understanding how different components of a predictive analytics workflow fit together. Before this project, I had limited experience with tools such as Jupyter Notebooks, Streamlit, pandas, NumPy, and XGBoost, and building an end-to-end pipeline—from data collection to deployment—required significant study, experimentation and troubleshooting.

To support my learning, I used a mixture of course material, internet searches, technical documentation and community discussions to understand errors, improve my approach and validate best practices. I have been careful not just to copy solutions, but to understand each step, document issues thoroughly, and apply the concepts independently across the project.

My real-world background in Business Intelligence, forecasting, and KPI-driven dashboard reporting helped greatly on the visualisation and communication side of the project. I regularly work with data in a commercial context, so translating model output into stakeholder-focused insights felt familiar. However, the machine learning aspects—model selection, training, tuning, and evaluation—were completely new, and represented a steep but rewarding challenge.

Overall, this project reflects both my technical learning journey and my practical experience in transforming data into meaningful, actionable insights.

The dashboard is built in Streamlit, powered by tabular, synthetic but realistic booking data, with models implemented in scikit-learn and XGBoost.

---

## 📚 Table of Contents

1. [Business Understanding (CRISP-DM)](#1-business-understanding-crisp-dm)  
   1.1 [Project Context](#11-project-context)  
   1.2 [Business Problem & Objectives](#12-business-problem--objectives)  
   1.3 [Stakeholders & Intended Users](#13-stakeholders--intended-users)  
   1.4 [Key Performance Indicators (KPIs)](#14-key-performance-indicators-kpis)  
   1.5 [Where the Image Model Fits in CRISP-DM](#15-where-the-image-model-fits-in-crisp-dm)  
   1.6 [Spatial Demand Overview (Map View)](#16-spatial-demand-overview-map-view)

2. [Business Requirements & User Stories](#2-business-requirements--user-stories)  
   2.1 [User Stories](#21-user-stories)  
   2.2 [Business Requirements](#22-business-requirements)  
   2.3 [Mapping Requirements to Visualisations & ML Tasks](#23-mapping-business-requirements-to-visualisations--ml-tasks)  

3. [Project Hypotheses & Validation Plan](#3-project-hypotheses--validation-plan)  

4. [Data & Features](#4-data--features)  
   4.1 [Data Sources](#41-data-sources)  
   4.2 [Targets](#42-targets)  
   4.3 [Feature Set](#43-feature-set)  

5. [Modelling & Evaluation](#5-modelling--evaluation)  
   5.1 [ML Business Case – Weekly Bookings Forecast (Regression)](#51-ml-business-case--weekly-bookings-forecast-regression)  
   5.2 [ML Business Case – Cancellation Risk (Classification)](#52-ml-business-case--cancellation-risk-classification)  
   5.3 [Model Training & Hyperparameter Tuning](#53-model-training--hyperparameter-tuning)  
   5.4 [Model Performance Summary](#54-model-performance-summary)  
   5.5 [Image Based Weather Severity Classifier](#55-image-based-weather-severity-classifier-supporting-model)

6. [Dashboard Design & Data Visualisation](#6-dashboard-design--data-visualisation)  
   6.1 [Dashboard Pages](#61-dashboard-pages)  
   6.2 [Key Plots & Interpretations](#62-key-plots--interpretations)  

7. [Deployment & Repository Structure](#7-deployment--repository-structure)  
   7.1 [Running the App Locally](#71-running-the-app-locally)  
   7.2 [Heroku Deployment](#72-heroku-deployment)  
   7.3 [Repository Structure](#73-repository-structure)  

8. [Main Libraries & Tools Used](#8-main-libraries--tools-used)  

9. [Troubleshooting & Technical Debugging Log](#9-troubleshooting--technical-debugging-log)  

10. [Unfixed Bugs](#10-unfixed-bugs)  

11. [Ethical Considerations](#11-ethical-considerations)  

12. [Future Improvements](#12-future-improvements)  
      [Learning Reflection & Development Journey](#learning-reflection--development-journey)  

13. [Learning Outcomes Mapping (LO1–LO7)](#13-learning-outcomes-mapping-lo1lo7)  

14. [Credits](#14-credits)  

---

## 1. Business Understanding (CRISP-DM)

### 1.1 Project Context

This project focuses on **winter mountain tours in the UK** (e.g. Lake District, Snowdonia, Scottish Highlands), where conditions are highly seasonal and heavily influenced by weather.

The tour provider needs to:

- Plan **guide staffing**, transport and equipment several weeks in advance.
- Manage **cancellations and no-shows** when conditions change.
- Keep tours safe while still operating profitably during peak and off-peak seasons.

The app is built as a **predictive analytics prototype** to support operational and commercial decisions, using historical-like bookings data combined with calendar and weather-derived features.

This corresponds to the **Business Understanding** phase of **CRISP-DM** and addresses **LO1.1**.

This section corresponds to the Business Understanding stage of CRISP-DM, where we define the context, objectives and success criteria for the ML tasks.

---

### 1.2 Business Problem & Objectives

**Key business problems:**

- **Demand uncertainty** – weekly bookings fluctuate across regions due to seasonality, bank holidays, school holidays and weather.
- **Cancellations and no-shows** – late cancellations lead to wasted guide capacity and under-utilised slots.
- **Resource planning** – guides must be rostered ahead of time; over-staffing is costly and under-staffing is risky.

**Main objectives:**

1. **Forecast weekly bookings per region**  
   To help Operations plan guide rosters, vehicle usage and maximum group sizes.

2. **Predict cancellation risk for individual bookings**  
   To allow Customer Service to target reminder communications and contingency planning for high-risk bookings.

3. **Quantify calendar and weather impact**  
   To understand how bank holidays, school holidays, peak winter, and severe weather conditions relate to demand and cancellations.

4. **Provide an interactive dashboard for non-technical users**  
   To allow Operations, Marketing and Customer Service to explore patterns and model outputs without needing to run notebooks.

---

### 1.3 Stakeholders & Intended Users

- **Operations Lead**  
  Needs weekly demand forecasts per region and clarity on peaks/lows.

- **Guides / Route Leaders**  
  Want visibility of expected tour volumes and typical conditions in their region.

- **Marketing & Sales**  
  Want to know when tours are under-booked so they can plan promotions or pricing changes.

- **Customer Service**  
  Needs cancellation risk scores to prioritise outbound reminders or rebooking offers.

- **Management / Business Owners**  
  Want evidence that modelling adds value and see the link between data, forecasts and operational decisions.

The Streamlit dashboard is designed for **non-technical users**, with clear narrative text and interpretations next to plots to support decisions (LO3.1, LO3.2).

---

### 1.4 Key Performance Indicators (KPIs)

To measure whether the predictive analytics solution is useful:

**Regression model (bookings forecast)**

- **MAE** (Mean Absolute Error)  
- **MAPE** (Mean Absolute Percentage Error)  
- **R²** (explained variance)

**Classification model (cancellation risk)**

- **ROC AUC**  
- **Recall** at a chosen operating threshold (focus: catching high-risk cancellations)  
- **Confusion matrix** to understand false positives vs false negatives

**Operational impact (conceptual in this prototype)**

- % of weeks where staffing is aligned with forecast  
- No-show / late cancellation rate on high-risk weeks after customer outreach

These metrics are reported in the modelling notebooks and surfaced in narrative form on the **Model Report** dashboard page (LO3.2, LO4.2, LO5.2).

### 1.5 Where the Image Model Fits in CRISP-DM

The Weather Severity Classifier functions as a parallel modelling stream outside the main CRISP-DM pipeline.

- It has its own small “Data Understanding → Modelling → Evaluation” cycle.

- It does not feed into the regression or classification pipelines.

- It complements the dashboard by offering an additional data modality (images), which strengthens interpretability and user engagement.

This mirrors real-world analytics workflows where secondary ML tools (e.g., photo analysis for conditions) support a larger operational system without altering its core prediction paths.


### 1.6 Spatial Demand Overview (Map View)

The Map View provides a spatial overview of forecasted weekly bookings across UK regions, translating numerical forecasts into a geographic format that is easier to interpret at a glance.

Rather than replacing the tabular or time-series forecasts, the map is designed to support rapid situational awareness for operational decision-makers. By visualising demand geographically, stakeholders can quickly identify:

- Regions under high demand pressure for a selected week

- Regions with lower expected demand where capacity may be reallocated

- How demand shifts week-by-week as the selected time period changes

Marker size and colour intensity represent forecasted booking volume, allowing non-technical users to assess relative demand without interpreting model outputs directly.

This aligns with the Business Understanding and Data Understanding stages of CRISP-DM by ensuring that model outputs are presented in a format that supports real operational decisions, such as guide allocation, transport planning, and regional prioritisation.

The Map View therefore acts as a decision-support visual layer, complementing the regression model and dashboard KPIs rather than introducing a new predictive task.

---

## 2. Business Requirements & User Stories

### 2.1 User Stories

**Operations Lead**

- As an Operations Lead, I want a **weekly bookings forecast per region**, so that I can roster the right number of guides in advance.
- As an Operations Lead, I want to **see the impact of holidays and weather on demand**, so that I can anticipate peak and low periods.

**Guides / Route Leaders**

- As a Guide, I want to **see expected tour volumes in my region**, so that I can plan my availability and travel.

**Marketing**

- As a Marketing Manager, I want to **identify weeks where demand is below normal**, so that I can plan promotions or discounts.
- As a Marketing Manager, I want to **understand which factors drive bookings**, so that I can design campaigns around key events (e.g. holidays).

**Customer Service**

- As a Customer Service agent, I want a **cancellation risk score for each upcoming booking**, so that I can prioritise reminder emails or calls for high-risk tours.

---

### 2.2 Business Requirements

From the user stories, the key business requirements are:

- **BR1 – Weekly bookings forecast per region**  
  The system must provide a forecast of total weekly bookings for each region.

- **BR2 – Cancellation risk prediction**  
  The system must provide a probability that a given booking will be cancelled, based on attributes like region, lead time and weather severity.

- **BR3 – Explain drivers of demand and cancellations**  
  The dashboard must show how calendar features and weather relate to bookings and cancellations.

- **BR4 – Interactive dashboard for non-technical users**  
  The solution must expose the models via a Streamlit dashboard with clear navigation, user inputs and plain-language explanations.

- **BR5 – Image-Based Weather Interpretation**  
  This feature enhances situational awareness and demonstrates how image-based ML models can complement tabular predictions.
  It does not influence forecasting or cancellation outputs but enriches decision support and user engagement.
---

### 2.3 Mapping Business Requirements to Visualisations & ML Tasks

| BR  | Description                               | Visualisations (EDA / Dashboard)                                                                 | ML Task(s)                                                                |
|-----|-------------------------------------------|---------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| BR1 | Weekly bookings forecast per region       | Time series plots by region; week × region heatmap; holiday vs non-holiday comparison           | **Regression** predicting weekly bookings with calendar, weather & lags   |
| BR2 | Cancellation risk prediction              | Cancellation rate by region/season; cancellation rate by weather severity                        | **Classification** predicting `was_cancelled` at booking level            |
| BR3 | Explain drivers of demand & cancellations | Feature importance plots; holiday uplift bars; cancellation vs weather severity plots            | Model feature importance / coefficients; optional SHAP-style interpretation |
| BR4 | Interactive dashboard for stakeholders    | Streamlit pages with region pickers, threshold sliders, and narrative text under plots           | Integration of trained models into a multipage Streamlit app              |

This satisfies the “rationale to map business requirements to data visualisations and ML tasks” requirement (LO2.1, LO2.2, LO3.2).

In summary, BR1 is answered by the weekly bookings regression model, and BR2 is answered by the cancellation classification model. BR3 is supported by both the EDA plots and model feature importance analysis, while BR4 is satisfied by the deployed Streamlit dashboard that exposes these models to non-technical users

---

## 3. Project Hypotheses & Validation Plan

The project defines the following hypotheses to deepen understanding of the data and model behaviour. These support Merit and Distinction criteria.

### H1: Holidays increase bookings in winter regions

> **Hypothesis H1:** School and bank holiday periods significantly increase bookings in core regions (e.g. Lake District, Snowdonia).

**Validation:**

- Compare mean weekly bookings in holiday weeks vs non-holiday weeks.
- Plot weekly bookings by region with holiday periods highlighted.
- Inspect holiday-related feature importance in the regression model.

**Outcome (summary):**

On the synthetic dataset, holiday weeks show higher average bookings than non-holiday weeks for key regions, and holiday indicators appear among the more influential features in the regression model.  
This **supports H1**.

---

### H2: Severe weather increases cancellations independent of season

> **Hypothesis H2:** Severe weather conditions (e.g. high wind, heavy precipitation, poor visibility) are associated with higher cancellation rates, even after accounting for season.

**Validation:**

- Group bookings by weather severity bins and compute cancellation rates.
- Plot cancellation rate vs weather severity.
- Inspect classification feature importance for weather-derived features.

**Outcome (summary):**

In the synthetic data, cancellation rate increases with weather severity. Weather-related features contribute to the classification model’s predictive power.  
This **supports H2**.

---

### H3: Lagged demand is the strongest single predictor of next week’s bookings

> **Hypothesis H3:** Lag features (e.g. bookings at t-1 and t-52) provide more predictive power for weekly bookings than any single calendar or weather feature.

**Validation:**

- Compare regression model performance with and without lag features.
- Inspect feature importance ranking for the final regression model.
- Qualitatively assess how removal of lag features changes MAE/R².

**Outcome (summary):**

In the tuned regression model, lag features (e.g. bookings at t-1 and t-52) rank highly in feature importance. Removing them increases error noticeably, indicating they are strong predictors of next week’s bookings.  
This **supports H3**.

These hypotheses, validation steps, and conclusions meet Merit criteria **1.2, 2.3, 4.3** and Distinction guidance on hypotheses.

---

### H4: Why Include Image-Based Weather Classification? (Real-World Relevance)

Mountain tour operators frequently rely on visual inspection of route conditions — snow coverage, wind on ridgelines, cloud base, visibility, or storm build-up.
This prototype demonstrates how ML could assist with:

- Rapid triage of user-submitted images
- Early detection of potentially hazardous conditions
- Supporting communications between guides and operations teams

While not used for automated decision-making, the feature illustrates how computer vision could enhance situational awareness in future versions of the Winter Mountain Tours platform.

---

## 4. Data & Features

### 4.1 Data Sources

The project uses a combination of **synthetic data** (for bookings, cancellations and weather) and **real endpoint data** (for bank holidays):

- **Bookings dataset (synthetic, tabular)**  
  Simulated booking-level records with region, route characteristics, party size, lead time and cancellation flag. Aggregated to weekly region-level for regression.

- **Cancellation dataset (synthetic)**  
  Derived from the bookings dataset with realistic cancellation patterns influenced by weather and lead times.

- **Weather dataset (synthetic)**  
  Weekly aggregates (e.g. temperature, precipitation, wind, visibility) with a derived **weather severity bin** to reflect hazard levels.

- **UK Bank Holidays endpoint (real)**  
  Data collected from the UK government bank holidays API using a dedicated Jupyter notebook (`00_collect_endpoint_bank_holidays.ipynb`).  
  This satisfies LO7.1 (data collection from an endpoint).

- **Calendar features (engineered)**  
  Derived from dates: year, week number, month, weekday distribution, “is peak winter” flag (Dec–Mar), and holiday indicators.

All datasets are clearly saved under `data/raw/`, `data/interim/` and `data/processed/` as per the project structure.

Dataset sizes (synthetic):

- Weekly regression dataset: ~N rows (weeks × regions), ~M features

- Booking-level classification dataset: ~K rows, M2 features

- Regions: Lake District, Snowdonia, Scottish Highlands

- Timeframe: 12 months of simulated winter-focused data

(PLACEHOLDER - FILL IN NUMBERS)

---

### 4.2 Targets

- **Regression target:**  
  `bookings_count` – weekly number of bookings per region.

- **Classification target:**  
  `was_cancelled` – binary label (1 = booking cancelled, 0 = not cancelled).

---

### 4.3 Feature Set

Key features include:

- **Region** (categorical)  
- **Week start date** and derived fields (year, week number, month)  
- **Calendar flags**: `is_bank_holiday_week`, `is_peak_winter`, simple school holiday indicator  
- **Weather variables**: mean temperature, precipitation, wind, visibility, and **weather severity bin**  
- **Lag features**: prior week bookings (`lag_1`), seasonal lag (`lag_52`), rolling averages  
- **Booking-level attributes** (for classification): region, party size, lead time, etc.

These features are created in `01_cleaning.ipynb` and `02_feature_engineering.ipynb`, meeting **LO7** and **Merit 7.2** for data preparation.

---

## 5. Modelling & Evaluation

### 5.1 ML Business Case – Weekly Bookings Forecast (Regression)

- **Aim**  
  Predict the total number of bookings per region per week to support capacity and staffing planning.

- **Learning method**  
  Supervised regression using historical weekly bookings and derived features.

- **Inputs (features)**  
  - Region  
  - Calendar features (week number, month, holiday flags, is_peak_winter)  
  - Aggregated weather features (temperature, precipitation, wind, visibility, severity bin)  
  - Lag features (t-1, t-52, rolling means)

- **Output (target)**  
  - `bookings_count` – integer/float representing bookings in a given region-week.

- **Success metrics**  
  - Primary: Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) on the test set.  
  - Secondary: R² on test set and inspection of Actual vs Predicted and residual plots.

- **Relevance for the user**  
  Even with a noisy, synthetic dataset, this demonstrates how an operations team could use a weekly forecast to inform guide rostering, transport and capacity decisions.

---

### 5.2 ML Business Case – Cancellation Risk (Classification)

- **Aim**  
  Predict the probability that an individual booking will be cancelled.

- **Learning method**  
  Supervised binary classification.

- **Inputs (features)**  
  - Region  
  - Lead time (days between booking date and tour date)  
  - Party size  
  - Calendar features for the tour date (week number, month, holiday flags, is_peak_winter)  
  - Weather severity bin for the tour date or week

- **Output (target)**  
  - `was_cancelled` – binary label (1 = cancelled, 0 = attended).

- **Success metrics**  
  - Primary: ROC AUC on the test set.  
  - Operational: Recall at a chosen decision threshold (prioritising capturing as many true cancellations as possible among flagged high-risk bookings).  
  - Supporting: Confusion matrix and classification report.

- **Relevance for the user**  
  A cancellation risk score enables Customer Service to focus on high-risk tours with reminders, offering rebooking options or checking conditions in advance.

---

### 5.3 Model Training & Hyperparameter Tuning

Two main pipelines were developed:

- **Bookings forecast (regression)**:  
  XGBoost Regressor used as final model, with experiments also run with Linear Regression and Random Forest.

- **Cancellation risk (classification)**:  
  XGBoost Classifier as final model, with Logistic Regression and Random Forest as baselines.

For the tuned XGBoost models, a grid search was conducted using at least **six hyperparameters** with **three or more values each**, e.g.:

- `n_estimators`  
- `max_depth`  
- `learning_rate`  
- `subsample`  
- `colsample_bytree`  
- `min_child_weight`  
- `reg_alpha` / `reg_lambda`

The ranges were chosen based on:

- Guidance from course material (e.g. cross-validation notebooks)
- XGBoost documentation
- Typical ranges used in similar tabular problems

This satisfies **Merit criterion 5.7** and the Distinction requirement for advanced modelling on tabular data.

---

## 5.4 Model Performance Summary

This project was developed iteratively. During development, the **synthetic data generation logic** was refined to better reflect real-world constraints (for example, enforcing non-negative bookings and introducing more realistic regional demand levels).  
As a result, **model performance improved significantly**, not due to changes in modelling complexity, but due to improvements in **data realism and stability**.

---

### Regression — Weekly Bookings Forecast (Tuned XGBoost)

#### Initial Results (Early Synthetic Dataset)

| Metric | Value |
|------|------|
| **MAE** | ~32.61 bookings |
| **MAPE** | ~119.64% |
| **R²** | ~–5.02 |

In the early version of the synthetic dataset, weekly bookings could take **negative or near-zero values** due to excessive noise relative to base demand.  
This led to unstable percentage errors and a negative R², despite the modelling pipeline itself being correctly implemented.

These results were documented transparently in the notebooks and on the **Model Report** page to demonstrate honest evaluation and reflection during development.

---

#### Updated Results (Refined Synthetic Dataset)

After improving the synthetic data generation to:

- Enforce **non-negative bookings**
- Introduce **region-specific base demand levels**
- Reduce excessive random noise
- Add additional realistic regions (Peak District, Yorkshire Dales)

the same modelling pipeline produced the following results:

| Model | MAE | MAPE | R² |
|-----|----|-----|---|
| Baseline (Lag-1) | ~14.7 | ~16.5% | ~0.51 |
| Linear Regression | ~9.1 | ~10.3% | ~0.81 |
| Random Forest | ~10.0 | ~11.3% | ~0.77 |
| **XGBoost (Tuned)** | **~9.4** | **~10.7%** | **~0.80** |

These results indicate:

- Forecast errors of roughly **±10 bookings per region per week**
- Predictions typically within **~10% of actual values**
- The model explains approximately **80% of weekly demand variation**

**Does the regression model meet the business requirement?**  
Yes. With more realistic synthetic data, the regression model provides forecasts accurate enough to support **capacity planning, guide rostering, and operational decision-making** at a regional level.

---

### Classification — Cancellation Risk (Tuned XGBoost)

| Model | ROC AUC |
|-----|--------|
| Logistic Regression | ~0.625 |
| Random Forest | ~0.564 |
| **XGBoost (Tuned)** | **~0.626** |

The cancellation model demonstrates **moderate discriminatory power**, performing consistently above random guessing (0.5).

**Does the classification model meet the business requirement?**  
Yes, as a **prototype**. The model is suitable for:

- Flagging higher-risk bookings
- Prioritising customer service outreach
- Supporting early-stage operational risk awareness

Given the synthetic nature of the data and the behavioural complexity of cancellations, a ROC AUC in the low-to-mid 0.6 range is realistic and acceptable for this stage.

---

### Key Learning & Interpretation

- The largest performance gains came from **improving data realism**, not increasing model complexity.
- This mirrors real-world analytics workflows, where **data quality often matters more than algorithm choice**.
- The project demonstrates:
  - End-to-end machine learning pipelines
  - Proper evaluation and comparison of multiple models
  - Honest reflection on limitations and subsequent improvements

All metrics, plots, and interpretations are presented in the notebooks and surfaced clearly on the **Model Report** dashboard page, aligning with **LO4.2** and **LO5.2**.

---

### Note on Tests & Validation

No automated tests rely on fixed performance thresholds, so no tests were broken by these changes.  
All model evaluation outputs were regenerated by re-running the modelling notebooks after updating the data generation logic.


**System Architecture Overview**

```

                 ┌────────────────────────┐
                 │      User (UI)         │
                 │ ─────────────────────── │
                 │  - Select region/week   │
                 │  - Enter booking data   │
                 │  - Upload mountain foto │
                 └─────────────┬──────────┘
                               │
                     Streamlit Front-End
                               │
       ┌───────────────────────┼────────────────────────┐
       │                       │                        │
       ▼                       ▼                        ▼
┌──────────────┐     ┌────────────────┐       ┌──────────────────────┐
│ Regression   │     │ Classification │       │ Weather Image Model  │
│ Forecasting  │     │ Cancellation  │       │  (Mild/Mod/Severe)    │
│ Model (XGB)  │     │ Risk (XGB)    │       │                      │
└─────┬────────┘     └───────┬───────┘       └───────────┬──────────┘
      │                      │                           │
      │                      │                           │
      ▼                      ▼                           ▼
 Weekly forecast     Cancellation score          Weather severity label
 for each region     for each booking           + class probabilities
```

---

 ### 5.5 Image-Based Weather Severity Classifier (Supporting Model)

In addition to the main regression and classification models, the project includes a prototype image-based weather severity classifier.
This model is designed as an auxiliary tool that demonstrates how image data can be incorporated into mountain safety or tour planning workflows.

**Learning Method**

A lightweight supervised classification model trained on simple image-derived features:

- brightness
- colourfulness
- proportion of bright pixels
- saturation/contrast proxies

These interpretable features are extracted from uploaded images using PIL and NumPy.
The model was implemented using scikit-learn and trained using labelled examples representing mild, moderate, and severe mountain conditions.

**Outputs**

Given an uploaded image, the model predicts:

- label – mild / moderate / severe
- class probabilities – adds transparency and insight into model confidence

**Purpose in the Project**

While not part of the main forecasting or cancellation pipelines, the image classifier:

Demonstrates end-to-end use of an image ML workflow within the same deployed app

Shows integration of different data modalities (tabular + image)

Supports a more contextual understanding of weather conditions

Enhances dashboard interactivity and interpretability

It is explicitly designed as a safe, isolated prototype that cannot affect the primary predictive tasks, aligning with good ML engineering practice.

---

## 6. Dashboard Design & Data Visualisation

### 6.1 Dashboard Pages

The Streamlit dashboard is implemented as a **multi-page app** under `app_pages/`:

1. **Home**  
   - Project overview  
   - Business context and high-level explanation of ML tasks  
   - Guidance on how to use the dashboard

2. **EDA & Insights**  
   - Seasonal trends in bookings  
   - Holiday uplift plots  
   - Cancellation rates vs weather severity  
   - Region comparisons and correlations  
   - Textual interpretations beneath each plot

3. **Bookings Forecast**  
   - Inputs: region, week start date, scenario toggles (bank holiday, peak winter, weather severity bin)  
   - Output: forecast bookings for that week + summary of input features used  
   - Explanation of model performance (MAE / MAPE / R²) and limitations

4. **Cancellation Risk**  
   - Inputs: booking-level attributes (e.g. region, party size, lead time, weather severity scenario)  
   - Output: cancellation probability and risk label (e.g. low/moderate/high)  
   - Threshold slider concept discussed in narrative

5. **Model Report**  
   - Summary of regression and classification metrics  
   - ROC and residuals plots  
   - Brief explanation of whether models meet their intended purpose

6. **Data & Docs**  
   - Data dictionary and feature descriptions  
   - Links to notebooks  
   - High-level explanation of data lineage (raw → processed)

7. **Weather from Image (Image-Based Severity Classifier)**  
   - Upload an image (JPG/PNG/WebP) showing mountain conditions 
   - View the uploaded image directly in the dashboard  
   - Run the Weather Severity Classifier to predict: severity label/class probabilities for transparency
   - Read plain-language interpretation text explaining what each severity category means

This page-level breakdown fulfils **LO6.1**.

**Business requirement coverage:**

BR1 (weekly bookings forecast) is mainly answered by the Bookings Forecast and Model Report pages.

BR2 (cancellation risk) is answered by the Cancellation Risk and Model Report pages.

BR3 (drivers and insights) is answered by the EDA & Insights and Model Report pages.

BR4 (interactive dashboard) is covered by the multi-page Streamlit layout with user inputs, selectors and explanatory text.

---

### 6.2 Key Plots & Interpretations

Examples of key visualisations and interpretations:

- **Seasonality line plot of weekly bookings by region**  
  - Interpretation: highlights peak winter weeks and holiday uplifts.

- **Holiday vs non-holiday weekly bookings bar plot**  
  - Interpretation: confirms the uplift hypothesis (H1).

- **Cancellation rate vs weather severity**  
  - Interpretation: supports H2 by showing increased cancellations under severe conditions.

- **Correlation heatmap / feature importance charts**  
  - Interpretation: shows relative influence of lag features, calendar flags and weather on predictions (supporting H3).

- **Interactive Plotly charts (e.g. region-filtered time series and bar charts)**  
  - Interpretation: allow users to explore different regions and time windows dynamically, supporting BR1 and BR3.

In the dashboard, each of these plots includes 1–2 lines of interpretation text directly underneath, explaining what the user is seeing and how it relates to the business questions (fulfilling LO6.2). Several EDA plots are implemented with Plotly to provide interactive tooltips and region selection, fulfilling the Merit 6.4–6.5 criteria.

Each plot is accompanied by **plain-language interpretation text** in the EDA and Model Report pages, addressing **LO6.2** and ensuring plots are not presented without context.

---

## 7. Deployment & Repository Structure

### 7.1 Running the App Locally

**Prerequisites:**

- Python 3.11+  
- Virtual environment (recommended)

**Steps:**

```bash
# Clone repository
git clone <repo-url>
cd winter-mountain-tours-demand-predictor

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

```
---

### 7.2 Heroku Deployment

The app is configured for deployment to Heroku using:

-   Procfile -- defines the web process running Streamlit.

-   setup.sh -- creates the Streamlit config in the Heroku environment.

-   requirements.txt -- curated list of Python packages (no heavy GPU libraries).

-   runtime.txt -- pins the Python version.

Deployment steps (high-level):

1.  Commit code and push to GitHub.

2.  Create a Heroku app and connect it to the GitHub repo.

3.  Enable automatic or manual deploy.

4.  Heroku builds the slug and runs the Streamlit app as defined in Procfile.

Several deployment issues were resolved along the way (e.g. slug too large, XGBoost version mismatches, missing data files); these are documented fully in the Troubleshooting & Technical Debugging Log.

---

### 7.3 Repository Structure

```

project/
│  README.md
│  Procfile
│  requirements.txt
│  runtime.txt
│  setup.sh
│  app.py
│
├─ app_pages/
│   ├─ page_home.py
│   ├─ page_eda.py
│   ├─ page_forecast.py
│   ├─ page_cancellation.py
│   └─ page_report.py
│
├─ src/
│   ├─ data_loaders.py
│   ├─ features.py
│   ├─ models.py
│   ├─ evaluate.py
│   └─ ui.py
│
├─ models/
│   ├─ bookings_model.pkl
│   └─ cancel_model.pkl
│
├─ data/
│   ├─ raw/
│   ├─ interim/
│   └─ processed/
│
├─ notebooks/
│   ├─ 00_collect_endpoint_bank_holidays.ipynb
│   ├─ 01_cleaning.ipynb
│   ├─ 02_feature_engineering.ipynb
│   ├─ 03_model_regression.ipynb
│   └─ 04_model_classification.ipynb
│
└─ reports/
    └─ figures/

```

**Version Control**

    This project was developed using Git and hosted on GitHub, with regular commits for each feature or fix. Branches were used for larger changes (e.g. modelling, Streamlit pages, deployment tweaks), and the commit history documents the evolution of the data pipeline, models and dashboard. This satisfies LO3.3 (use of Git & GitHub for version control).

---

## 8. Main Libraries & Tools Used

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF9900?style=for-the-badge&logo=xgboost&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-0B3D91?style=for-the-badge)
![requests](https://img.shields.io/badge/requests-2C5BB4?style=for-the-badge)
![Pillow](https://img.shields.io/badge/Pillow-3670A0?style=for-the-badge&logo=pillow&logoColor=white)


- **Streamlit** – user interface/dashboard for ML prototyping. Used to build the multipage layout, user input widgets and display plots.
- **pandas** – data loading, cleaning, aggregation and feature engineering.
- **NumPy** – numerical computations and array operations.
- **scikit-learn** – train/test split, baseline models (Logistic Regression, Random Forest), metrics and pipeline utilities.
- **XGBoost** – final gradient boosting models for regression and classification.
- **matplotlib & seaborn** – EDA plots and evaluation charts (e.g. ROC curves, correlation heatmaps).
- **Plotly** – interactive visualisations in the EDA dashboard.
- **joblib** – saving/loading trained models (`.pkl` files).
- **requests** – accessing the UK bank holidays endpoint in `00_collect_endpoint_bank_holidays.ipynb`.
- **Pillow (PIL)** – used to load and process image files for the Weather Severity Classifier (e.g., resizing, color extraction).



Example usage:

-   Seaborn used to draw a correlation heatmap to explore relationships between numerical features.

-   Scikit-learn's `roc_auc_score` used to compute ROC AUC for the classification model.

-   XGBoost used to train an ensemble model with tuned hyperparameters for bookings forecasting.

This satisfies the "main data analysis and machine learning libraries" README requirement.

---

9\. Troubleshooting & Technical Debugging Log
---------------------------------------------

A full breakdown of issues encountered during development, their causes, and resolutions is provided here to demonstrate end-to-end debugging capability across Python, Jupyter, ML tooling, and cloud deployment.

### 1\. Jupyter Notebooks Failing to Run

**Symptoms**

`ModuleNotFoundError: No module named 'requests'
ModuleNotFoundError: No module named 'sklearn'
ModuleNotFoundError: No module named 'xgboost'`

**Cause**

Packages were not installed inside the active virtual environment.

**Fix**

-   Verified interpreter used by Jupyter:

    `python -c "import sys; print(sys.executable)"`

-   Installed missing dependencies in the venv:

    `pip install requests scikit-learn xgboost`

* * * * *

### 2\. Missing Calendar Columns & Feature Engineering Breakages

**Symptoms**

`KeyError: ['year', 'week_number', 'month', ...] not in index`

**Cause**

Earlier notebooks (e.g. data collection / feature engineering) had not been run in sequence, so processed datasets were incomplete and expected columns were missing.

**Fix**

Re-ran notebooks **in order**:

1.  `00_collect_endpoint_bank_holidays.ipynb`

2.  `01_cleaning.ipynb`

3.  `02_feature_engineering.ipynb`

4.  `03_model_regression.ipynb`

5.  `04_model_classification.ipynb`

This regenerated all processed CSVs with the full feature set.

* * * * *

### 3\. Processed Data Missing in Deployment

**Symptoms (on Heroku)**

`FileNotFoundError: /app/data/processed/weekly_bookings_regression.csv`

**Cause**

`.gitignore` was initially excluding:

`data/interim/
data/processed/`

As a result, processed data was never pushed to Heroku.

**Fix**

Updated `.gitignore`:

`!data/processed/
!data/processed/*.csv`

Committed processed data and redeployed.

* * * * *

### 4\. Heroku Deployment Failure: Slug Too Large

**Symptoms**

`Compiled slug size: 653.1M is too large (max is 500M)`

**Cause**

An earlier `pip freeze` captured heavy GPU-related packages (CUDA, `nvidia-nccl`, GPU-enabled XGBoost builds), drastically increasing slug size.

**Fix**

Replaced the auto-generated `requirements.txt` with a **hand-curated**, minimal list:

`streamlit==1.40.1
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
xgboost==1.7.6
plotly==5.22.0
matplotlib==3.8.4
seaborn==0.13.2`

Slug size dropped well below Heroku's limit and deployment succeeded.

* * * * *

### 5\. Missing Plotting Libraries in Production

**Symptoms**

`ModuleNotFoundError: No module named 'matplotlib'`

**Cause**

The local environment had `matplotlib` and `seaborn` installed, but they were missing from `requirements.txt`.

**Fix**

Added:

`matplotlib==3.8.4
seaborn==0.13.2`

Redeployed to Heroku; EDA and evaluation plots rendered correctly.

* * * * *

### 6\. XGBoost Attribute Errors (Model Loading Issues)

**Symptoms (various pages)**

`AttributeError: 'XGBModel' object has no attribute 'gpu_id'
AttributeError: 'XGBModel' object has no attribute 'predictor'
AttributeError: 'XGBClassifier' object has no attribute 'use_label_encoder'`

**Cause**

Models were trained locally with a newer XGBoost version (e.g. 3.x), but Heroku used `xgboost==1.7.6`.\
Pickle files contain all model attributes, including those not recognised by older versions.

**Fix (Two-Part)**

1.  Standardised version:

    `pip install xgboost==1.7.6
    python -c "import xgboost; print(xgboost.__version__)"  # verify`

    Re-trained both regression and classification models under 1.7.6.

2.  Defensive attribute removal in `src/data_loaders.py` before returning the estimator:

    `for attr in ["gpu_id", "predictor", "use_label_encoder"]:
        if hasattr(est, attr):
            delattr(est, attr)`

This prevents legacy attributes from causing failures and provides resilience for future retrains.

* * * * *

### 7\. Timestamp Slice Error on Forecast Page

**Symptoms**

`TypeError: 'Timestamp' object is not subscriptable`

Triggered by:

`inputs["week_start"][:10]`

**Cause**

`inputs["week_start"]` is a `pandas.Timestamp`, which cannot be sliced like a string.

**Fix**

Replaced with:

`inputs["week_start"].strftime("%Y-%m-%d")`

This formats the timestamp safely for display.

* * * * *

### 8\. Final Deployment Stabilisation

Additional actions taken:

-   Rebuilt models under the standardised XGBoost version.

-   Ensured processed datasets are committed and available in production.

-   Cleaned unused imports and simplified Streamlit config for compatibility.

-   Smoke-tested all pages on Heroku (Home, EDA, Forecast, Cancellation, Model Report, Data & Docs).

* * * * *

### Summary of Key Lessons Learned

| Issue Category | Lesson Learned |
| --- | --- |
| Environment | Always verify the interpreter for Jupyter & avoid `pip freeze` bloat |
| Data pipeline | Downstream steps depend on running upstream notebooks in order |
| Deployment | Slug size and dependencies must be curated |
| Serialization | ML models are sensitive to library version mismatches |
| Debugging | Trace errors to the correct layer (data, model, UI, deployment) |
| Streamlit/Heroku | Keep config simple and libraries lightweight |

This log demonstrates real-world ML engineering skills beyond basic modelling.

---

10\. Unfixed Bugs
-----------------

At the time of submission:

-   There are **no known unfixed bugs** that break core functionality of the Streamlit app or notebooks.

-   Minor visual or UX refinements (e.g. additional plot polishing or layout tweaks) are possible but do not affect the learning outcomes.

If any issues are discovered after submission, they will be documented in the repository's issue tracker.

---

11\. Ethical Considerations
---------------------------

Although this project uses **synthetic data**, a real-world deployment would need to consider:

-   **Safety implications**\
    Forecasts and cancellation risk should complement, not replace, professional judgement about mountain safety. Severe weather decisions must remain with qualified leaders.

-   **Fairness and transparency**\
    Cancellation risk scores must not be used to penalise customers unfairly. Instead, they should support better communication (e.g. reminders, clear policies).

-   **Data privacy**\
    Real booking data would require compliance with GDPR and privacy regulations. Identifiable information should be minimised and secured.

-   **Model limitations**\
    Models trained on limited or biased data can misrepresent risk. The dashboard includes narrative text to highlight that this is a **prototype** and that model performance should be continuously monitored and improved.

    ---

12\. Future Improvements
------------------------

Possible next steps:

-   Integrate a **real weather API** (e.g. Met Office or OpenWeather) for more realistic features.

-   Implement **SHAP** or other explainability methods directly in the dashboard.

-   Extend forecasting to **route-level** rather than only region-level.

-   Add **multi-model comparison** options (e.g. choose between XGBoost, Random Forest, etc.).

-   Introduce **alerting** (e.g. email notifications) for high-cancellation-risk tours.

-   Improve calibration of the classification model and tighten its decision thresholds.

-   Add a larger data set (with more variety) for the image based weather predictor to improve accuracy 

-   Sales based dashboards which show turnover, Profit & Loss accounts, margins etc

---

## Learning Reflection & Development Journey


This project represented a significant learning curve, particularly in applying machine learning concepts end-to-end for the first time.
While I had prior professional experience working with BI tools, KPIs, and dashboarding for operational decision-making, the modelling,
feature engineering, and evaluation aspects of predictive analytics were entirely new.

To develop this project, I relied on a combination of:
- Course material and walkthrough projects,
- Official library documentation (e.g. pandas, scikit-learn, XGBoost, Streamlit),
- Targeted internet searches to troubleshoot errors and deepen understanding,
- Iterative experimentation through Jupyter notebooks.

This mirrors real-world analytics work, where engineers and analysts frequently consult documentation and external resources
to solve problems and validate approaches. All modelling logic, feature choices, and interpretations in this project reflect
my own understanding and implementation.

The project therefore demonstrates both technical learning and the practical problem-solving process required to build
a working predictive analytics system from raw data through to a deployed dashboard.


---

13\. Learning Outcomes Mapping (LO1--LO7)
----------------------------------------

**LO1 -- Fundamentals of AI/ML/Data Science**

-   Business Understanding and dataset description covered in [Sections 1--4].

**LO2 -- Frame ML Problems to Produce Value**

-   User stories, business requirements, and mapping to ML tasks in [Section 2].

**LO3 -- Business Drivers & Data Science Effectiveness**

-   Stakeholder analysis and ML Business Cases in [Sections 1.3, 5.1, 5.2].

**LO4 -- Obtain Actionable Insights**

-   EDA and interpretations throughout [Section 6]; operational implications discussed in [Sections 3, 5, 11].

**LO5 -- Create Intelligent Systems Using ML**

-   Models, evaluation, and pipeline described in [Section 5] and implemented in the notebooks and Streamlit app. In addition to forecasting/cancellation models the project implements an image-based weather classifier

**LO6 -- Represent Data Stories via Visualisation**

-   Dashboard design, visualisations and explanations in [Section 6]. The image-based weather predictor provides additional visual story around weather conditions

**LO7 -- Collect, Arrange & Process Data**

-   Endpoint collection in `00_collect_endpoint_bank_holidays.ipynb`, data cleaning and feature engineering in `01_cleaning.ipynb` and `02_feature_engineering.ipynb`.

Each outcome is supported by concrete code, notebooks, and deployed app behaviour.

---

14\. Credits
------------

-   **Code Institute** -- for the Predictive Analytics learning material, walkthrough projects (Churnometer, Malaria Detector, Heritage Housing, Cherry Leaves) and Streamlit/CRISP-DM structure that inspired this project's design.

-   **UK Government Bank Holidays API** -- for the real bank holidays data used in the calendar features.

-   **XGBoost, scikit-learn, pandas, NumPy, Streamlit, matplotlib, seaborn** -- open-source libraries that made the analysis and modelling possible.

-   **Google Images** used creative commons licence for some imagery for AI traning, however most images are my own

-   **Chat GPT** was used to support with readme structuring & layout

-   Any other referenced documentation or blog posts are cited through comments in the notebooks where relevant.