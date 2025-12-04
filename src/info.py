# src/info.py

DATA_FIELDS = {
    "region": "Tour region (categorical)",
    "bookings_count": "Weekly bookings target",
    "was_cancelled": "Cancellation target (0/1)",
    "week_start": "Start date of week",
    "year": "Calendar year",
    "week_number": "ISO week number",
    "month": "Month number (1–12)",
    "is_bank_holiday_week": "Binary flag for UK holiday week",
    "is_peak_winter": "Binary winter peak flag",
    "mean_temp_c": "Avg temp (°C)",
    "precip_mm": "Avg precipitation (mm)",
    "snowfall_flag": "Snow observed (0/1)",
    "weather_severity_bin": "Weather severity level",
    "lag_1w_bookings": "Previous week's bookings",
    "lag_4w_mean": "Rolling 4-week mean",
    "lag_52w_bookings": "Bookings last year same week",
    "party_size": "Booking group size",
    "lead_time_days": "Lead time from booking to tour date",
    "route_difficulty": "Tour difficulty rating",
}
