import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 25)
RNG = 42

# --- Plot styling
CAT = {
        "blue": "#2a78d6", 
        "orange": "#eb6834", 
        "aqua": "#1baf7a", 
        "yellow": "#eda100",
        "magenta": "#e87ba4", 
        "green": "#008300", 
        "violet": "#4a3aa7", 
        "red": "#e34948"
    }
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

########### Initial Data Check #################

# for dataset
DATA_DIR = './Part1/data'

# convert dteday from string to date datatype
day = pd.read_csv(f"{DATA_DIR}/day.csv", parse_dates=["dteday"])
hour = pd.read_csv(f"{DATA_DIR}/hour.csv", parse_dates=["dteday"])

print("day.csv  :", day.shape)
print("hour.csv  :", hour.shape)

print("\nDay Data Count == Casual + Registered :", (day['cnt'] == day['casual'] + day['registered']).all())
print("Hour Data Count == Casual + Registered :", (hour['cnt'] == hour['casual'] + hour['registered']).all())

# Summary Stats from pandas
day.describe().T
# print(day.describe().T)

# CHECHKING DATA ANOMALIES

# mISSING dAYS
total_expected_hours = hour['dteday'].nunique() * 24
actual_hours = len(hour)
missing_hours_cnt = total_expected_hours - actual_hours
counts_per_day = hour.groupby('dteday').size()
incomplete_days = (counts_per_day < 24).sum()

print(f"\nhour.csv contains {len(hour)} of {total_expected_hours} expected hourly records "
      f"-> {missing_hours_cnt} missing hours ({missing_hours_cnt/total_expected_hours*100:.1f}%), "
      f"distributed in {incomplete_days} of {len(counts_per_day)} days.")

# Zero Humidity
hum_zero = hour.loc[hour['hum'] == 0]
hum_zero = hour.loc[hour['hum'] == 0]
print(f"\nhum == 0% anomaly: {len(hum_zero)} consecutive hours, all on "
      f"{hum_zero['dteday'].dt.date.unique()} -> most likely an anomaly.")

# Extreme weather count
print(f"\nweathersit == 4 (severe storm) hours: {(hour['weathersit']==4).sum()} of {len(hour)} "
      f"-> too rare to support a reliable coefficient on its own"
      f"can lead to inaccuracy Task 2/3 models.")

# print(f"\nwindspeed == 0 hours: {(hour['windspeed']==0).sum()} ({(hour['windspeed']==0).mean()*100:.1f}%) "
#       f"-> plausibly true calm-wind readings, but also consistent with an instrument floor/rounding "
#       f"artifact; flagged as an open data-provenance question rather than assumed benign.")

# Daily-count outlier scan (Tukey IQR rule) -- sanity check for extreme days
Q1, Q3 = day['cnt'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outlier_days = day[(day['cnt'] < lo) | (day['cnt'] > hi)]
print(f"\nDaily-count IQR outlier bounds: [{lo:.0f}, {hi:.0f}] -> {len(outlier_days)} outlier days at the daily "
      f"level (the series is well-behaved once aggregated to daily totals; extremes only emerge hour-by-hour).")


###### Task 1: Descriptive Statistics and Exploratory Analysis #########

# Growth from 2011 to 2012
total_years = day.groupby('yr')['cnt'].sum()
# print(total_years)
yoy = (total_years.iloc[1] / total_years.iloc[0] - 1) * 100
print(f"\n2011 total rentals: {total_years.iloc[0]:,} | 2012 total rentals: {total_years.iloc[1]:,} | YoY growth: {yoy:.1f}%")

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(day["dteday"], day["cnt"], color=CAT["blue"], lw=0.9, alpha=0.45, label="Daily count")
ax.plot(day["dteday"], day["cnt"].rolling(14, center=True).mean(), color=CAT["blue"], lw=2.2, label="14-day rolling mean")
ax.set_title("Daily Total Rentals, Jan 2011 – Dec 2012"); ax.set_ylabel("Rentals per day")
ax.legend(loc="upper left")
plt.show()

# Hourly demand: Casual vs Registered on Working Days vs Weekends&Holidays
weekday = hour[hour["workingday"] == 1].groupby("hr")[["casual", "registered"]].mean()
weekend = hour[hour["workingday"] == 0].groupby("hr")[["casual", "registered"]].mean()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)

axes[0].plot(weekday.index, weekday["registered"], color=CAT["blue"], lw=2.2, label="Registered")
axes[0].plot(weekday.index, weekday["casual"], color=CAT["orange"], lw=2.2, label="Casual")
axes[0].set_title("Working Days"); axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("Average rentals")
axes[0].set_xticks(range(0, 24, 3)); axes[0].legend(loc="upper left")

axes[1].plot(weekend.index, weekend["registered"], color=CAT["blue"], lw=2.2, label="Registered")
axes[1].plot(weekend.index, weekend["casual"], color=CAT["orange"], lw=2.2, label="Casual")
axes[1].set_title("Weekends & Holidays"); axes[1].set_xlabel("Hour of day"); axes[1].set_xticks(range(0, 24, 3))

fig.suptitle("Hourly Demand Profile by User Type and Day Type", fontweight="bold")
plt.show()

print("Working-day registered peaks: hr=8 (", round(weekday.loc[8,'registered']), ") and hr=17 (", round(weekday.loc[17,'registered']), ")")
print("Non-working-day peak (both types): around hr=", weekend['registered'].idxmax(), "-", weekend['casual'].idxmax())

#Weather Data

weather_map = {
    1: "Clear/Few Clouds",
    2: "Mist/Cloudy",
    3: "Light Rain/Snow",
    4: "Heavy Rain/Snow"
}
wcat = hour.copy(); wcat["weather_label"] = wcat["weathersit"].map(weather_map)
grp = (
            wcat.groupby("weather_label")["cnt"]
                  .agg(["mean", "std", "count"])
                  .reindex([weather_map[i] for i in [1,2,3,4]])
      )

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(grp.index, grp["mean"], color=SEQ_BLUE[3], width=0.6)
ax.errorbar(grp.index, grp["mean"], yerr=grp["std"]/np.sqrt(grp["count"]), fmt="none", ecolor=INK_SECONDARY, capsize=4)
ax.set_ylabel("Mean hourly rentals"); ax.set_title("Mean Hourly Rentals by Weather Condition (± SE)")
ax.set_xticks(range(len(grp.index))); ax.set_xticklabels(grp.index, rotation=12, ha="right")
plt.show()
print(grp)

#Casual vs Registered Bike Rides on season basis
season_map = {
    1: "Spring", 
    2: "Summer", 
    3: "Fall", 
    4: "Winter"
}
dcat = day.copy(); dcat["season_label"] = dcat["season"].map(season_map)
sgrp = dcat.groupby("season_label")[["casual", "registered"]].mean().reindex([season_map[i] for i in [1,2,3,4]])

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(sgrp.index, sgrp["registered"], color=CAT["blue"], label="Registered")
ax.bar(sgrp.index, sgrp["casual"], bottom=sgrp["registered"], color=CAT["orange"], label="Casual")
ax.set_ylabel("Mean daily rentals"); ax.set_title("Mean Daily Rentals by Season (Casual vs. Registered)")
ax.legend(loc="upper right")
plt.show()
sgrp.round(0)


# Correlation Matrix
# --- Correlation structure (daily aggregates) -------------------------
num_cols = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
corr = day[num_cols].corr()

fig, ax = plt.subplots(figsize=(6.2, 5.4))
ax.grid(False)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=45, ha="right")
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(corr.iloc[i,j]) > 0.55 else INK_PRIMARY)
ax.set_title("Correlation Matrix (Daily Aggregates)")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
plt.show()
print("temp vs. atemp correlation:", round(day['temp'].corr(day['atemp']), 3), "-> near-collinear; see Task 2.")

#casual riders over-time


# --- Casual-user demand share over time ---------------------------------
day["casual_share"] = day["casual"] / day["cnt"]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(day["dteday"], (day["casual_share"].rolling(14, center=True).mean())*100, color=CAT["aqua"], lw=2)
ax.set_ylabel("Casual share of rentals (%)"); ax.set_title("Casual-User Share of Total Rentals (14-day rolling mean)")
plt.show()
print(f"Overall casual share: {hour['casual'].sum()/hour['cnt'].sum()*100:.1f}% of all rentals across both years.")

