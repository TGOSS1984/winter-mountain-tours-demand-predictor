# Data Model – Winter Mountain Tour Demand & Cancellation Predictor

This project uses synthetic bookings data combined with real calendar features
(UK bank holidays) and simulated weather data.

## 1. Raw / Source Tables

### 1.1 `bookings_raw` (synthetic, booking-level)

Grain: **one row per booking**.

Fields:

- `booking_id` (int) – unique identifier for each booking.
- `region` (category) – region code, e.g. `"lake_district"`, `"snowdonia"`, `"highlands"`.
- `tour_date` (date) – date of the tour.
- `booking_date` (date) – date the booking was made.
- `party_size` (int) – number of people in the booking.
- `route_difficulty` (category) – e.g. `"easy"`, `"moderate"`, `"challenging"`.
- `was_cancelled` (0/1) – 1 if the booking was cancelled, otherwise 0.

### 1.2 `calendar_raw` (from endpoint + derived)

Grain: **one row per week per region** (after processing), but source may be
per holiday date.

Core fields (after processing):

- `week_start` (date) – Monday of the week.
- `week_number` (int) – ISO week number.
- `year` (int) – calendar year.
- `month` (int) – month number.
- `is_bank_holiday_week` (0/1) – week contains at least one UK bank holiday.
- `is_school_holiday_week` (0/1) – simple synthetic school holiday flag.
- `is_peak_winter` (0/1) – e.g. 1 for Dec–Mar.

### 1.3 `weather_weekly_raw` (synthetic)

Grain: **one row per region per week**.

Fields:

- `region` (category)
- `week_start` (date)
- `mean_temp_c` (float)
- `precip_mm` (float)
- `snowfall_flag` (0/1)
- `wind_speed_kph` (float)
- `visibility_km` (float)
- `weather_severity_bin` (category) – e.g. `"mild"`, `"moderate"`, `"severe"`.

## 2. Processed Tables for Modelling

### 2.1 `weekly_bookings` (for regression)

Grain: **one row per region per week**.

Fields (examples):

- `region`
- `week_start`
- `bookings_count` (int) – total bookings in the week (target for regression)
- Calendar features: `week_number`, `month`, `is_bank_holiday_week`, `is_school_holiday_week`, `is_peak_winter`
- Weather features: `mean_temp_c`, `precip_mm`, `snowfall_flag`, `wind_speed_kph`, `visibility_km`, `weather_severity_bin`
- Lag features (created later in feature engineering): 
  - `lag_1w_bookings`
  - `lag_52w_bookings`
  - `rolling_4w_mean`

### 2.2 `bookings_for_classification` (for cancellation model)

Grain: **one row per booking**.

Fields (examples):

- `booking_id`
- `region`
- `tour_date`
- `lead_time_days` – (tour_date - booking_date)
- `party_size`
- `route_difficulty`
- Calendar and weather features joined at the tour week
- Target: `was_cancelled` (0/1).

## 3. File Locations

- Raw data: `data/raw/`
  - `bookings_raw.csv`
  - `weather_weekly_raw.csv`
  - `bank_holidays_raw.json` (from endpoint)
- Interim data: `data/interim/`
  - `calendar_weeks.csv`
  - `bookings_cleaned.csv`
- Processed data for modelling: `data/processed/`
  - `weekly_bookings_regression.csv`
  - `bookings_for_classification.csv`
