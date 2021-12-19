# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Conda (ML)
#     language: python
#     name: conda-ml
# ---

# %%
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# %matplotlib inline
plt.rcParams["figure.figsize"] = (15, 7)

# %%
covid = pd.read_csv("https://data.cityofnewyork.us/resource/rc75-m7u3.csv")

covid["date_of_interest"] = covid.date_of_interest.astype("datetime64[ns]")
covid.set_index("date_of_interest", inplace=True)
# throw out data before May 2020, since testing was inconsistent and hospitalizations/deaths were very high
covid = covid.loc[date.fromisoformat("2020-05-01"):]
# also the last day seems super unreliable
covid = covid.iloc[:-1]
covid = covid.copy()
covid["day_of_week"] = covid.index.weekday


# %%
# this wasn't useful but I'll keep it around just in case
def rolling_winsorize(s, frac_to_clip, window, **kwargs):
    rolling = s.rolling(window, **kwargs)
    lower_clip = rolling.quantile(frac_to_clip, interpolation="lower")
    upper_clip = rolling.quantile(1 - frac_to_clip, interpolation="higher")
    return s.clip(lower_clip, upper_clip)


# %%
def adjust_by_weekday(df, data_col, weekday_col):
    log_mean_by_weekday = np.log(df[data_col].replace(0, 1)).groupby(df[weekday_col]).mean()
    correction_factor = np.exp(log_mean_by_weekday - log_mean_by_weekday.mean()).rename("__correction_factor")
    with_correction = df.join(correction_factor, on=weekday_col)
    return with_correction[data_col] / with_correction["__correction_factor"]


# %%
cases_adj = adjust_by_weekday(covid, "case_count", "day_of_week").rename("cases_adj")
hosp_adj = adjust_by_weekday(covid, "hospitalized_count", "day_of_week").rename("hosp_adj")

# %%
covid_adj = covid[["case_count", "hospitalized_count", "death_count"]].join(cases_adj).join(hosp_adj)

# %%
# there's a large nonzero mean and an obvious 1-year-frequency effect that I don't care about
# so remove 5 components on each end
plt.plot(np.abs(np.fft.fft(covid_adj.case_count)[5:-5]))

# %%
# this makes it clear that there is some weekday effect in hospitalizations, it's just not as strong as in counts
plt.plot(np.abs(np.fft.fft(covid_adj.hospitalized_count)[5:-5]))

# %%
covid_adj[["hospitalized_count", "hosp_adj"]].plot()
# The adjustment really isn't any smoother. That's odd. No point using it.

# %%
log_mean_by_weekday = np.log(covid.case_count.replace(0, 1)).groupby(covid.day_of_week).mean()
case_correction_factor = np.exp(log_mean_by_weekday - log_mean_by_weekday.mean()).rename("case_weekday_correction")

# %%
covid_adj = covid[["case_count", "hospitalized_count", "death_count", "day_of_week"]].join(case_correction_factor, on="day_of_week")
covid_adj["cases_adj"] = covid_adj.eval("case_count / case_weekday_correction")
covid_adj.cases_adj.plot()
# this still has a pretty notable day-of-week effect but it's way less extreme than in the original data

# %%
covid_adj[["cases_adj", "case_count", "hospitalized_count", "death_count"]].plot()

# %%
cases_trailing_week = covid["case_count"].rolling(7).sum()
hosp_trailing_week = covid["hospitalized_count"].rolling(7).sum()
# early 2020 was so bad for NYC and testing was so rare that it's washing out the rest
(hosp_trailing_week / cases_trailing_week).loc[date.fromisoformat("2020-05-01"):].plot()

# %%
# should be able to predict hospitalizations as a function of recent case counts, and deaths as a function of recent hospitalizations
# linear regression seems almost certainly correct, since there should be a slowly-time-varying response function from infection to hospitaliation
# perhaps likewise from hospitalization to death
# need to account for different scales of data
# also need to account for the highly correlated predictors. A prior that adjacent coefficients are equal seems helpful.

# %%
# fan shape - classic example of heteroskedasticity
plt.scatter("cases_adj", "hospitalized_count", data=covid_adj)


# %%
def regress_squeeze_adjacent(X, y, weight, lam):
    """Regularized linear regression y ~ X beta
    Adjacent predictors are regularized to have similar coefficients"""
    n, p = X.shape
    if weight is None:
        weight = np.ones(n)
    xtx = np.einsum("ij,ik,i", X, X, weight)
    for i in range(p - 1):
        fake_obs = np.zeros(p)
        fake_obs[i] = 1.
        fake_obs[i+1] = -1.
        xtx += lam * np.outer(fake_obs, fake_obs)
    xty = np.einsum("ij,i,i", X, y, weight)
    return np.linalg.solve(xtx, xty)


# %%
case_to_hospitalized = covid_adj[["cases_adj", "hospitalized_count"]].copy()
max_hospitalized_lag = 14
predictors = []
for i in range(1, max_hospitalized_lag):
    pred = f"cases_lagged_{i}"
    case_to_hospitalized[pred] = case_to_hospitalized.cases_adj.shift(i)
    predictors.append(pred)

case_to_hospitalized.dropna(inplace=True)
pred_norm_sq = (case_to_hospitalized[predictors] ** 2).sum(axis=1)
row_var_estimator = pred_norm_sq.clip(lower=pred_norm_sq.median())
case_to_hospitalized["weight"] = 1 / row_var_estimator

# %%
plt.scatter(case_to_hospitalized.eval("cases_adj * weight ** 0.5"), case_to_hospitalized.eval("hospitalized_count * weight ** 0.5"))
# a significant iprovement, though still somewhat heteroskedastic

# %%
shifts = range(-20, 20)
corrs = [case_to_hospitalized.cases_adj.shift(pred_shift).corr(case_to_hospitalized.hospitalized_count) for pred_shift in shifts]

# %%
# first fit an autoregressive model to cases - expect to see momentum
coef_autoreg = pd.Series(
    regress_squeeze_adjacent(
        case_to_hospitalized[predictors],
        case_to_hospitalized.cases_adj,
        case_to_hospitalized.weight,
        1e8 * case_to_hospitalized.weight.mean(),
    ),
    index=predictors,
)
plt.plot(coef_autoreg)
_ = plt.xticks(rotation=50)

# %%
plt.plot(shifts, corrs)  # peak at around 9 days lagged
# possibility: when cases spike, the people who test positive are more responsible
# seeing as they're getting tested. THen later come the hospitalizations, which
# happen after a few days mostly to people who get exposed and don't get tested.
# Probably a correlation between not getting vaccinated and not getting tested.

# %%
first_half = case_to_hospitalized.iloc[:300]
coefs = pd.Series(
    regress_squeeze_adjacent(
        first_half[predictors],
        first_half.hospitalized_count,
        first_half.weight,
        1e8 * first_half.weight.mean(),
    ),
    index=predictors,
)
plt.plot(coefs)
_ = plt.xticks(rotation=50)

# %%
second_half = case_to_hospitalized.iloc[300:]
coefs = pd.Series(
    regress_squeeze_adjacent(
        second_half[predictors],
        second_half.hospitalized_count,
        second_half.weight,
        1e8 * second_half.weight.mean(),
    ),
    index=predictors,
)
plt.plot(coefs)
_ = plt.xticks(rotation=50)

# %% [markdown]
# This is a rather oddly shaped graph. Higher coefficients on older data is particularly weird.

# %%
# add some exponentially decaying time weight, halflife = 8 months, so we can include all the data
time_weight = 2 ** (np.arange(len(case_to_hospitalized)) / 240.0)
total_weight = case_to_hospitalized.weight * time_weight

final_coefs = pd.Series(
    regress_squeeze_adjacent(
        case_to_hospitalized[predictors],
        case_to_hospitalized.hospitalized_count,
        total_weight,
        1e8 * total_weight.mean(),
    ),
    index=predictors,
)
plt.plot(final_coefs)
_ = plt.xticks(rotation=50)

# %%
predictions = case_to_hospitalized[predictors] @ final_coefs

# %%
hosp_diff = pd.DataFrame({"actual_hospitalized": case_to_hospitalized.hospitalized_count, "predicted_hospitalized": predictions})
hosp_diff["residual"] = hosp_diff.eval("actual_hospitalized - predicted_hospitalized")

# %%
hosp_diff.plot()
# residual is super noisy. Probably day-of-week-related but it was hard to correct for

# %%
# perhaps 100 fewer hospitalizations per day than expected? That's a huge difference.
# Even after some conservative smoothing it's 50ish fewer
hosp_diff.residual.plot()
hosp_diff.residual.rolling(7).mean().plot()

# %%
