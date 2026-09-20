import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import matplotlib.pyplot as plt

RNG = 42

# for dataset
DATA_DIR = './Part1/data'

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

hour2 = pd.read_csv(f"{DATA_DIR}/hour.csv")
hour2["timestamp"] = pd.to_datetime(hour2["dteday"]) + pd.to_timedelta(hour2["hr"], unit="h")
hour2 = hour2.set_index("timestamp").sort_index()

full_idx = pd.date_range(hour2.index.min(), hour2.index.max(), freq="h")
full = hour2.reindex(full_idx); full.index.name = "timestamp"
print(f"Reindexed to a complete hourly grid: {len(full)} hours total, {full['cnt'].isna().sum()} originally-missing hours now explicit NaN.")

calendar_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
weather_cols = ["temp", "atemp", "hum", "windspeed"]
full[calendar_cols] = full[calendar_cols].ffill()
full[weather_cols] = full[weather_cols].ffill()
full["hr"] = full.index.hour
full["weekday"] = full.index.dayofweek.map({0:1,1:2,2:3,3:4,4:5,5:6,6:0})  # Sunday = 0
full["day_of_month"] = full.index.day

# --- Feature engineering strategy --------------------

## encoding cyclic calendar values
full["hr_sin"] = np.sin(2*np.pi*full["hr"]/24)
full["hr_cos"] = np.cos(2*np.pi*full["hr"]/24)

full["dow_sin"] = np.sin(2*np.pi*full["weekday"]/7)
full["dow_cos"] = np.cos(2*np.pi*full["weekday"]/7)

full["month_sin"] = np.sin(2*np.pi*full["mnth"]/12)
full["month_cos"] = np.cos(2*np.pi*full["mnth"]/12)

## cretaing features based on previous bike-demand values

full["lag_1h"]   = full["cnt"].shift(1)
full["lag_24h"]  = full["cnt"].shift(24)
full["lag_168h"] = full["cnt"].shift(168)   # 1 week earlier 24x7=168

#use only the previous value
#The .shift(1): exclude current hour to help prevent data leakage
full["roll24_mean"] = full["cnt"].shift(1).rolling(24, min_periods=12).mean()       #recent 24 hr evg
full["roll7d_mean"] = full["cnt"].shift(1).rolling(24*7, min_periods=24*3).mean()   #recent 7 d evg
full["roll24_std"]  = full["cnt"].shift(1).rolling(24, min_periods=12).std()        #recent 24h variability

model_df = full.loc[full["cnt"].notna()].copy()
model_df["train"] = model_df["day_of_month"] <= 20
print(f"Modeling frame: {len(model_df)} real hourly observations "
      f"({model_df['train'].sum()} train / {(~model_df['train']).sum()} test).")

lag_cols = ["lag_1h", "lag_24h", "lag_168h", "roll24_mean", "roll7d_mean", "roll24_std"]
train_medians = model_df.loc[model_df["train"], lag_cols].median()
n_imputed = model_df[lag_cols].isna().sum().sum()
model_df[lag_cols] = model_df[lag_cols].fillna(train_medians)
print(f"Imputed {n_imputed} remaining NaN lag-feature cells (edge effect: first ~7 days of the series lack a "
      f"168h history) using TRAIN-only medians.")

#model inputs
cal_weather_feats = ["season", "yr", "holiday", "workingday", "weathersit", "temp", "hum", "windspeed",
                      "hr_sin", "hr_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
lag_feats = ["lag_1h", "lag_24h", "lag_168h", "roll24_mean", "roll7d_mean", "roll24_std"]
all_feats = cal_weather_feats + lag_feats

X_train, X_test = model_df.loc[model_df["train"], all_feats], model_df.loc[~model_df["train"], all_feats]
y_train, y_test = model_df.loc[model_df["train"], "cnt"], model_df.loc[~model_df["train"], "cnt"]
print(f"\n{len(all_feats)} features: {all_feats}")

#####  EVALUATIO METRICS ############
"""
    y_true  : actual bike-demand values
    y_pred  : predicted values
    name    :   Model name

"""
def metrics(y_true, y_pred, name, store):
    y_pred = np.clip(y_pred, 0, None)           #replace negative predictions with 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:32s} RMSE={rmse:7.2f}  MAE={mae:7.2f}  MAPE={mape:6.2f}%  R2={r2:.4f}")
    store.append(dict(model=name, RMSE=rmse, MAE=mae, MAPE=mape, R2=r2))
    return y_pred

results = []

# --- Model 1: climatology baseline (hour x workingday mean, TRAIN only) ----
clim = model_df.loc[model_df["train"]].groupby(["hr", "workingday"])["cnt"].mean()
pred_clim = model_df.loc[~model_df["train"]].apply(
    lambda r: clim.get(
        (r["hr"], r["workingday"]), 
        y_train.mean()), 
    axis=1
)
_ = metrics(y_test.values, pred_clim.values, "Baseline: climatology", results)


# --- Model 2: exogenous-only Negative Binomial GLM (no AR lags) ------------
# hour-of-day enters as a CATEGORICAL dummy here, not sin/cos: Task 1/2 showed the
# true daily curve is sharply bimodal, which a single-harmonic sinusoid cannot
# represent in a log-linear model (it forces a smooth, unimodal shape). Categorical
# dummies give the GLM the same nonparametric flexibility in time-of-day as the
# baseline and the tree models, so this is a fair "best a linear model can do" comparison.
glm_feats = ["hr", "season", "yr", "holiday", "workingday", "weathersit", "temp", "hum", "windspeed"]
Xg_train = pd.get_dummies(model_df.loc[model_df["train"], glm_feats], columns=["hr","season","weathersit"], drop_first=True).astype(float)
Xg_test = pd.get_dummies(model_df.loc[~model_df["train"], glm_feats], columns=["hr","season","weathersit"], drop_first=True).astype(float)
Xg_test = Xg_test.reindex(columns=Xg_train.columns, fill_value=0.0)
Xg_train_c = sm.add_constant(Xg_train); Xg_test_c = sm.add_constant(Xg_test, has_constant="add")

nb_glm = dm.NegativeBinomial(y_train.values, Xg_train_c.values).fit(disp=False, maxiter=200)
pred_nb = nb_glm.predict(Xg_test_c.values)
_ = metrics(y_test.values, pred_nb, "GLM (NegBin, exogenous-only)", results)


# --- Model 3: Random Forest, exogenous-only (isolates model-class effect) --
rf_exog = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_leaf=3, random_state=RNG, n_jobs=-1)
rf_exog.fit(X_train[cal_weather_feats], y_train)
pred_rf_exog = rf_exog.predict(X_test[cal_weather_feats])
_ = metrics(y_test.values, pred_rf_exog, "Random Forest (exogenous-only)", results)

# --- Model 4: Random Forest, full feature set (calendar + weather + AR lags)
rf_full = RandomForestRegressor(n_estimators=400, max_depth=18, min_samples_leaf=2, random_state=RNG, n_jobs=-1)
rf_full.fit(X_train, y_train)
pred_rf_full = rf_full.predict(X_test)
_ = metrics(y_test.values, pred_rf_full, "Random Forest (+ AR lags)", results)

# --- Model 5: XGBoost, full feature set -------------------------------
xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=RNG, n_jobs=-1)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
pred_xgb_clipped = metrics(y_test.values, pred_xgb, "XGBoost (+ AR lags)", results)

results_df = pd.DataFrame(results)
results_df

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(results_df["model"], results_df["RMSE"], color=SEQ_BLUE[3])
axes[0].set_xlabel("RMSE (rentals/hour)"); axes[0].set_title("Test RMSE by Model")
axes[1].barh(results_df["model"], results_df["MAPE"], color=CAT["orange"])
axes[1].set_xlabel("MAPE (%)"); axes[1].set_title("Test MAPE by Model")
fig.tight_layout(); plt.show()

# --- Performance across the 24 monthly evaluation windows -----------------
test_eval = model_df.loc[~model_df["train"]].copy()
test_eval["pred"] = np.clip(pred_xgb, 0, None)
test_eval["yr_mnth"] = test_eval.index.to_period("M").astype(str)

def rmse_grp(g): return np.sqrt(mean_squared_error(g["cnt"], g["pred"]))
def mape_grp(g): return np.mean(np.abs((g["cnt"]-g["pred"])/np.clip(g["cnt"],1,None)))*100

monthly_perf = test_eval.groupby("yr_mnth").apply(
    lambda g: pd.Series({"RMSE": rmse_grp(g), "MAPE": mape_grp(g), "mean_actual": g["cnt"].mean(), "n_hours": len(g)}),
    include_groups=False)

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(len(monthly_perf)), monthly_perf["RMSE"], color=SEQ_BLUE[3])
ax.set_xticks(range(len(monthly_perf))); ax.set_xticklabels(monthly_perf.index, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("RMSE (rentals/hour)"); ax.set_title("XGBoost Test RMSE Across the 24 Monthly Evaluation Windows")
fig.tight_layout(); plt.show()
monthly_perf.round(2)

for m, label in [("2011-02", "Feb 2011 (low season)"), ("2012-07", "Jul 2012 (peak season)")]:
    sub = test_eval.loc[test_eval["yr_mnth"] == m]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(sub.index, sub["cnt"], color=CAT["blue"], lw=1.3, label="Actual")
    ax.plot(sub.index, sub["pred"], color=CAT["red"], lw=1.3, ls="--", label="XGBoost prediction")
    ax.set_title(f"Test-Window Forecast vs. Actual — {label} (days 21–end)")
    ax.set_ylabel("Hourly rentals"); ax.legend(loc="upper left")
    fig.tight_layout(); plt.show()


# --- Feature importance & calibration --------------------------------
importances = pd.Series(xgb_model.feature_importances_, index=all_feats).sort_values()
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.barh(importances.index, importances.values, color=SEQ_BLUE[3])
ax.set_title("XGBoost Feature Importance (Gain-based)"); ax.set_xlabel("Relative importance")
fig.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(test_eval["cnt"], test_eval["pred"], s=6, alpha=0.25, color=CAT["blue"])
lims = [0, max(test_eval["cnt"].max(), test_eval["pred"].max())]
ax.plot(lims, lims, color=INK_SECONDARY, lw=1.3, ls="--")
ax.set_xlabel("Actual hourly count"); ax.set_ylabel("Predicted hourly count")
ax.set_title("XGBoost: Predicted vs. Actual (Test Set)")
fig.tight_layout(); plt.show()