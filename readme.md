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

## 2. Business Requirements & User Stories

### 2.1 User Stories

**Operations Lead**

- As an Operations Lead, I want a **weekly bookings forecast per region**, so that I can roster the right number of guides in advance.
- As an Operations Lead, I want to **see the impact of holidays and weather on demand**, so that I can anticipate peak and low periods.

**Guides / Route Leaders**

- As a Guide, I want to **see expected tour volumes in my region**, so that I can plan my availability and travel.

**Marketing**

- As a Marketing Manager, I want to **identify weeks where demand is below normal**, so that I can plan promotions or discounts.
- As a Marketing Manager, I want to **understand which factors drive bookings**, so that I can tailor campaigns (e.g. around holidays).

**Customer Service**

- As a Customer Service agent, I want a **cancellation risk score for each upcoming booking**, so that I can prioritise reminder emails or calls for high-risk tours.

### 2.2 Business Requirements

From the user stories, the key business requirements are:

- **BR1 – Weekly bookings forecast per region**  
  The system must provide a forecast of total weekly bookings for each region (e.g. Lake District, Snowdonia, Highlands).

- **BR2 – Cancellation risk prediction**  
  The system must provide a probability that a given booking will be cancelled, based on attributes like region, lead time and forecast weather.

- **BR3 – Explain drivers of demand and cancellations**  
  The dashboard must show how calendar features (e.g. bank holidays, school holidays, peak winter) and weather conditions relate to bookings and cancellations.

- **BR4 – Interactive dashboard for non-technical users**  
  The solution must expose the models via an easy-to-use Streamlit dashboard with clear navigation, text explanations and plot interpretations.

### 2.3 Mapping Business Requirements to Visualisations & ML Tasks

The table below maps each business requirement to specific data visualisations and ML tasks that will be implemented in the notebooks and Streamlit app.

| BR | Description                               | Visualisations (EDA / Dashboard)                                          | ML Task(s)                                   |
|----|-------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------|
| BR1 | Weekly bookings forecast per region       | Time series plots of weekly bookings by region; heatmap of week × region; holiday vs non-holiday comparison | **Regression model** predicting weekly bookings per region using calendar, weather and lag features |
| BR2 | Cancellation risk prediction              | Cancellation rate by region and season; cancellation rate by weather bins | **Classification model** predicting `was_cancelled` at booking level |
| BR3 | Explain drivers of demand & cancellations | Correlation/feature importance plots; bar charts for holiday uplift; SHAP or feature importance charts | Use model feature importance / coefficients and partial dependence to interpret drivers of bookings and cancellations |
| BR4 | Interactive dashboard for stakeholders    | Streamlit pages with region filters, date/threshold sliders, and textual interpretations under each plot | Integration of trained models into a Streamlit app with user inputs and prediction outputs |

At least one ML task (regression for bookings and classification for cancellations) is explicitly defined and linked to the business requirements and visualisations, satisfying the rationale mapping requirement (LO2.1, LO2.2, LO3.2).


---

## 3. Project Hypotheses & Validation Plan  <!-- Merit, Distinction -->

*(To be completed in 1.5)*

---

## 4. Data & Features

*(Will be filled in Phase 2)*

---

## 5. Modelling & Evaluation

### 5.1 ML Business Case – Weekly Bookings Forecast (Regression)

- **Aim**  
  Predict the total number of bookings per region per week, to support capacity and staffing planning.

- **Learning method**  
  Supervised regression using historical weekly bookings with calendar, weather and lag features.

- **Inputs (features)**  
  - Region (categorical)
  - Week start date and derived calendar features (week number, month, is_peak_winter, bank holiday flags, school holiday flags)
  - Aggregated weather features (mean temperature, snowfall flag, precipitation, wind, visibility)
  - Lag features (e.g. bookings at t-1, t-52 and rolling averages)

- **Output (target)**  
  - `bookings_count` – number of bookings in the region for the given week.

- **Success metrics**  
  - Primary: MAE and MAPE on the test set.
  - Secondary: R² on test set and visual inspection of Actual vs Predicted and residual plots.

- **Relevance for the user**  
  The forecast allows the Operations Lead to roster guides and allocate capacity with less guesswork. Lower error means fewer weeks with over- or under-staffing.

### 5.2 ML Business Case – Cancellation Risk (Classification)

- **Aim**  
  Predict the probability that an individual booking will be cancelled, so that Customer Service can prioritise proactive communication and reminders.

- **Learning method**  
  Supervised binary classification using booking attributes, calendar and weather features.

- **Inputs (features)**  
  - Region
  - Lead time (days between booking date and tour date)
  - Party size
  - Calendar features (week number, month, holiday flags, is_peak_winter)
  - Forecast / typical weather features for the tour date (e.g. severe weather flags or bins)

- **Output (target)**  
  - Binary label `was_cancelled` (1 = cancelled, 0 = attended).

- **Success metrics**  
  - Primary: ROC AUC on test set.
  - Operational: Recall at a chosen decision threshold (focus on catching as many true cancellations as possible among high-risk bookings).
  - Supporting: Confusion matrix and classification report at the chosen threshold.

- **Relevance for the user**  
  The cancellation risk score enables targeted reminders (SMS/email) for high-risk bookings, which can reduce no-shows and improve customer experience.

Detailed modelling steps, hyperparameter tuning and evaluation plots will be implemented in dedicated Jupyter Notebooks (regression and classification). Each notebook will include an Objectives/Inputs/Outputs section at the top, as required by LO5.5.


---

## 6. Dashboard Design & Data Visualisation

*(Will be filled when Streamlit pages are in place)*

---

## 7. Deployment & Repository Structure

*(Will be filled at the end)*

---

## 8. Learning Outcomes Mapping

*(Final pass once everything else is written)*
