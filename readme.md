# Winter Mountain Tour Demand & Cancellation Predictor

A Streamlit-based predictive analytics app that forecasts weekly winter tour bookings
by region and predicts cancellation risk. Built for **Portfolio Project 5: Predictive Analytics**
on the Code Institute Full Stack Software Development Diploma.

> **Status:** Business understanding and requirements definition in progress.

---

## 1. Business Understanding (CRISP-DM)

### 1.1 Project Context

This project focuses on winter mountain tours in the UK (e.g. Lake District, Snowdonia, Scottish Highlands),
where conditions are highly seasonal and weather-dependent. The tour provider needs to plan guide staffing,
group capacity and customer communications several weeks in advance, while managing cancellations and no-shows
caused by severe weather or changing customer plans.

The app is built as a predictive analytics prototype to support operational and commercial decisions, using
historical bookings data combined with calendar and weather features.

### 1.2 Business Problem & Objectives

The key business problems are:

- **Demand uncertainty** – weekly bookings fluctuate across regions due to seasonality, holidays and weather.
- **Cancellations and no-shows** – late cancellations and no-shows create wasted guide capacity and poor customer experience.
- **Resource planning** – guides must be rostered and allocated to regions in advance, but over-staffing is costly.

From these problems, the main objectives are:

1. **Forecast weekly bookings per region** so Operations can plan guide capacity, vehicles and equipment.
2. **Predict cancellation risk for individual bookings** so the business can target reminders or follow-up for high-risk tours.
3. **Quantify the impact of calendar and weather factors** (e.g. bank holidays, school holidays, severe weather) on bookings and cancellations.
4. **Provide an interactive dashboard** that non-technical stakeholders can use to explore demand patterns and model outputs.

This description of the dataset and business requirements corresponds to the **Business Understanding** phase in CRISP-DM (LO1.1).

### 1.3 Stakeholders & Intended Users

- **Operations Lead** – needs accurate weekly forecasts per region to plan guide rosters and capacity.
- **Guides / Route Leaders** – need visibility of likely tour volumes to plan availability and travel.
- **Marketing & Sales** – want to identify peak weeks and low-demand periods to target promotions and adjust pricing.
- **Customer Service** – need cancellation risk flags to prioritise proactive communication and reminders.
- **Management** – want an overview of performance versus forecast and evidence that predictive models are reliable.

The Streamlit dashboard is designed for non-technical users, with clear narratives and plot interpretations to support decisions (LO3.1, LO3.2).

### 1.4 Key Performance Indicators (KPIs)

To evaluate whether the predictive analytics solution adds value, the project focuses on the following KPIs:

- **Forecast model KPIs (bookings regression)**
  - Mean Absolute Error (MAE) on test set.
  - Mean Absolute Percentage Error (MAPE) on test set.
  - R² on test set as a measure of explained variance.

- **Cancellation model KPIs (classification)**
  - ROC AUC on test set.
  - Recall at a chosen operating threshold (focus on catching high-risk cancellations).
  - Confusion matrix at the chosen threshold to understand trade-offs between false positives and false negatives.

- **Operational impact indicators (qualitative)**
  - % of weeks where staffing was aligned with forecast (on-time staffing).
  - No-show / late cancellation rate for high-risk weeks after potential communications.

The first two sets of KPIs will be computed directly from the models. The operational impact indicators are described qualitatively in this prototype and would be measured in a real deployment.

These KPIs will be used later to determine whether the ML pipeline has met the business requirements (LO3.2, LO4.2, LO5.2).


---

## 2. Business Requirements & User Stories  <!-- LO2 -->

### 2.1 User Stories

*(To be completed in 1.3)*

### 2.2 Business Requirements

*(To be completed in 1.3)*

### 2.3 Mapping BR → Visualisations & ML Tasks

*(To be completed in 1.3)*

---

## 3. Project Hypotheses & Validation Plan  <!-- Merit, Distinction -->

*(To be completed in 1.5)*

---

## 4. Data & Features

*(Will be filled in Phase 2)*

---

## 5. Modelling & Evaluation

*(Will be filled after modelling)*

---

## 6. Dashboard Design & Data Visualisation

*(Will be filled when Streamlit pages are in place)*

---

## 7. Deployment & Repository Structure

*(Will be filled at the end)*

---

## 8. Learning Outcomes Mapping

*(Final pass once everything else is written)*
