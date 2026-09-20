
import pandas as pd

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

import matplotlib.pyplot as plt
import numpy as np

import statsmodels.discrete.discrete_model as dm
from scipy import stats as sstats

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

# for dataset
DATA_DIR = './Part1/data'

day = pd.read_csv(f"{DATA_DIR}/day.csv", parse_dates=["dteday"])
###### Task 2: Statistical and Regression Analysis #########

day2 = day.copy()
day2["season"] = day2["season"].astype("category")
day2["weathersit"] = day2["weathersit"].astype("category")

# --- Multicollinearity screen (why atemp is dropped) -----------------------
cont = sm.add_constant(day2[["temp", "atemp", "hum", "windspeed"]])
vif = pd.DataFrame({"variable": cont.columns,
                     "VIF": [variance_inflation_factor(cont.values, i) for i in range(cont.shape[1])]})
print("VIF with both temp and atemp:\n", vif.to_string(index=False))

cont2 = sm.add_constant(day2[["temp", "hum", "windspeed", "yr", "holiday", "workingday"]])
vif2 = pd.DataFrame({"variable": cont2.columns,
                      "VIF": [variance_inflation_factor(cont2.values, i) for i in range(cont2.shape[1])]})
print("\nVIF after dropping atemp:\n", vif2.to_string(index=False))

# --- OLS baseline ------------------------------------------------------
formula = "cnt ~ C(season) + yr + holiday + workingday + C(weathersit) + temp + hum + windspeed"
ols = smf.ols(formula, data=day2).fit()
print(ols.summary())

# --- Residual diagnostics ------------------------------------------------
bp_stat, bp_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
jb_stat, jb_p, skew, kurt = jarque_bera(ols.resid)
dw = durbin_watson(ols.resid)
print(f"Breusch-Pagan (heteroskedasticity): LM={bp_stat:.2f}, p={bp_p:.4g}")
print(f"Jarque-Bera (normality): JB={jb_stat:.2f}, p={jb_p:.4g}, skew={skew:.2f}, kurtosis={kurt:.2f}")
print(f"Durbin-Watson (autocorrelation): {dw:.3f}  (2.0 = none; well below 2 => positive serial correlation)")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].scatter(ols.fittedvalues, ols.resid, s=10, color=CAT["blue"], alpha=0.5)
axes[0].axhline(0, color=INK_SECONDARY, lw=1); axes[0].set_xlabel("Fitted values"); axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs. Fitted")
sm.qqplot(ols.resid, fit=True, line="45", ax=axes[1], markerfacecolor=CAT["blue"], markeredgecolor="none", alpha=0.5)
axes[1].set_title("Normal Q-Q")
axes[2].plot(day2["dteday"], ols.resid, color=CAT["orange"], lw=0.9); axes[2].axhline(0, color=INK_SECONDARY, lw=1)
axes[2].set_title("Residuals over Time"); axes[2].tick_params(axis='x', labelrotation=30)
fig.tight_layout(); plt.show()

# --- Non-linearity check: is the temperature effect linear? ----------------
formula_q = "cnt ~ C(season) + yr + holiday + workingday + C(weathersit) + temp + I(temp**2) + hum + windspeed"
ols_q = smf.ols(formula_q, data=day2).fit()

f_val = ((ols.ssr - ols_q.ssr) / 1) / (ols_q.ssr / ols_q.df_resid)
f_p = 1 - sstats.f.cdf(f_val, 1, ols_q.df_resid)
print(f"temp coefficient: {ols_q.params['temp']:.1f} (p={ols_q.pvalues['temp']:.4g})")
print(f"temp^2 coefficient: {ols_q.params['I(temp ** 2)']:.1f} (p={ols_q.pvalues['I(temp ** 2)']:.4g})")
print(f"R2: linear={ols.rsquared:.4f}  vs.  +temp^2={ols_q.rsquared:.4f}")
print(f"F-test for the added quadratic term: F={f_val:.2f}, p={f_p:.4g}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(day2["temp"], day2["cnt"], s=10, color=CAT["blue"], alpha=0.4)
temp_grid = np.linspace(day2["temp"].min(), day2["temp"].max(), 100)
z = np.polyfit(day2["temp"], day2["cnt"], 2)
ax.plot(temp_grid, np.polyval(z, temp_grid), color=CAT["red"], lw=2.5, label="Quadratic fit")
ax.set_xlabel("Normalized temperature"); ax.set_ylabel("Daily rental count")
ax.set_title("Rentals vs. Temperature (Non-linear Relationship)"); ax.legend()
plt.show()


# --- Count-data distribution: Poisson vs. Negative Binomial ----------------
print(f"cnt: mean={day2['cnt'].mean():.1f}  variance={day2['cnt'].var():.1f}  "
      f"variance/mean ratio={day2['cnt'].var()/day2['cnt'].mean():.1f}  (>>1 => overdispersed)")

poisson_fit = smf.glm(formula, data=day2, family=sm.families.Poisson()).fit()
dispersion = poisson_fit.pearson_chi2 / poisson_fit.df_resid
print(f"Poisson GLM Pearson chi2/df (dispersion): {dispersion:.2f}  (>>1 => Negative Binomial preferred)")


# --- Negative Binomial regression --------------------------------------
X_design = pd.get_dummies(day2[["season", "weathersit"]], drop_first=True).astype(float)
X_design = pd.concat([X_design, day2[["yr", "holiday", "workingday", "temp", "hum", "windspeed"]].astype(float)], axis=1)
X_design = sm.add_constant(X_design)

nb_model = dm.NegativeBinomial(day2["cnt"], X_design).fit(disp=False)
print(nb_model.summary())
print("\nIncidence Rate Ratios (exp(coef)):")
print(np.exp(nb_model.params).round(3))


# --- Casual vs. registered: different behavioral drivers -------------------
nb_casual = dm.NegativeBinomial(day2["casual"], X_design).fit(disp=False)
nb_registered = dm.NegativeBinomial(day2["registered"], X_design).fit(disp=False)

compare = pd.DataFrame({
    "casual_coef": nb_casual.params, "casual_p": nb_casual.pvalues,
    "registered_coef": nb_registered.params, "registered_p": nb_registered.pvalues,
}).round(3)
print(compare)

print(f"\nworkingday IRR  — casual: {np.exp(nb_casual.params['workingday']):.3f}  "
      f"registered: {np.exp(nb_registered.params['workingday']):.3f}")
print(f"holiday IRR     — casual: {np.exp(nb_casual.params['holiday']):.3f}  "
      f"registered: {np.exp(nb_registered.params['holiday']):.3f}")
print(f"temp IRR        — casual: {np.exp(nb_casual.params['temp']):.3f}  "
      f"registered: {np.exp(nb_registered.params['temp']):.3f}")

vars_to_plot = [c for c in compare.index if c not in ("const", "alpha")]
yidx = np.arange(len(vars_to_plot))
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(yidx - 0.18, compare.loc[vars_to_plot, "casual_coef"], height=0.36, color=CAT["orange"], label="Casual")
ax.barh(yidx + 0.18, compare.loc[vars_to_plot, "registered_coef"], height=0.36, color=CAT["blue"], label="Registered")
ax.set_yticks(yidx); ax.set_yticklabels(vars_to_plot, fontsize=9)
ax.axvline(0, color=INK_SECONDARY, lw=1)
ax.set_xlabel("Negative Binomial coefficient (log scale)"); ax.set_title("Driver Comparison: Casual vs. Registered Demand")
ax.legend(); fig.tight_layout(); plt.show()