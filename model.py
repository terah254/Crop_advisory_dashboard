import pandas as pd
import numpy as np

def engineer_features(daily_forecast: list) -> pd.DataFrame:
    df = pd.DataFrame(daily_forecast)

    # Temperature features
    df["temp_range"]   = df["temp_max"] - df["temp_min"]
    df["heat_stress"]  = (df["temp_max"] > 32).astype(int)
    df["cold_stress"]  = (df["temp_min"] < 10).astype(int)
    df["optimal_temp"] = (
        (df["temp_max"] >= 20) & (df["temp_max"] <= 30)
    ).astype(int)

    # Rainfall features
    df["cumulative_rain"] = df["precipitation_sum"].cumsum()
    df["drought_flag"]    = (df["precipitation_sum"] < 1.0).astype(int)
    df["flood_risk_flag"] = (df["precipitation_sum"] > 30).astype(int)

    # Rain probability
    df["high_rain_prob"] = (df["precipitation_probability"] > 70).astype(int)

    # Wind
    df["high_wind"] = (df["wind_max"] > 40).astype(int)

    # Composite stress score
    df["crop_stress_score"] = (
        df["heat_stress"]      * 25 +
        df["cold_stress"]      * 20 +
        df["flood_risk_flag"]  * 20 +
        df["drought_flag"]     * 15 +
        df["high_rain_prob"]   * 10 +
        df["high_wind"]        * 10
    ).clip(0, 100)

    df["planting_window"] = (df["crop_stress_score"] < 20).astype(int)

    return df


def generate_advisory(df: pd.DataFrame, tree_data: dict, high_risk_days: int) -> list:
    tips = []

    if df["flood_risk_flag"].sum() > 0:
        tips.append("Heavy rain expected — delay fertilizer application to avoid run-off.")

    if df["drought_flag"].sum() >= 4:
        tips.append("Dry stretch detected — consider irrigation scheduling.")

    if df["heat_stress"].sum() > 2:
        tips.append("High temperatures forecast — increase irrigation frequency.")

    if df["high_rain_prob"].sum() > 3:
        tips.append("High chance of rain most days — hold off on spraying pesticides.")

    if tree_data:
        needs_care = tree_data.get("tree_health", {}).get("needs_care", 0)
        if needs_care > 5:
            tips.append(f"{needs_care} trees need attention — inspect for pests or disease.")

    if not tips:
        tips.append("Conditions look favorable this week. Good window for planting or top-dressing.")

    return tips