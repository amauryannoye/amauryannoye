
import pandas as pd
import numpy as np
import itertools
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import statsmodels.api as sm
import random

# === USER PARAMS ===
MAX_VARS = 10
MAX_LAG = 40
MAX_FEATURES = 150
CORR_THRESHOLD = 0.85
N_JOBS = -1  # use all CPU cores
MAX_LAG_COMBOS = 100  # max lag sets per variable set

TARGET_VAR = "Difference"
DATE_CUTOFF = "2019-11-24"
TARGET_DATES = {
    'first_zero':  "Jun-20",
    'max':         "Sep-20",
    'second_zero': "Mar-22",
    'min':         "Nov-22",
    'last_zero':   "Jan-25"
}

# === Load data ===
df = pd.read_excel('Variable_Data_Cleaned.xlsx', parse_dates=['Date'], index_col='Date')
df = df.asfreq('MS')
df[TARGET_VAR] = df['Spot (DAT)'] - df['Contract_DAT']

train = df[df.index < DATE_CUTOFF]
test = df[df.index >= DATE_CUTOFF]
combined = pd.concat([train, test])

all_vars = [col for col in df.columns if col not in ['Spot (DAT)', 'Contract_DAT', TARGET_VAR]]

# === Utility functions ===

def poly_zeros(poly):
    roots = np.roots(poly.coeffs)
    return sorted([r.real for r in roots if np.isreal(r)])

def poly_extrema(poly):
    d_poly = poly.deriv()
    crit_points = [r.real for r in np.roots(d_poly.coeffs) if np.isreal(r)]
    if not crit_points:
        return None, None
    crit_vals = [poly(r) for r in crit_points]
    return crit_points[np.argmax(crit_vals)], crit_points[np.argmin(crit_vals)]

def target_date_to_x(target_str, forecast_start):
    target_date = pd.to_datetime(target_str, format="%b-%y")
    return (target_date.year - forecast_start.year)*12 + (target_date.month - forecast_start.month)

def create_lagged_features(df, variables, lags):
    df_lagged = pd.DataFrame(index=df.index)
    for var in variables:
        for lag in lags:
            df_lagged[f"{var}_lag{lag}"] = df[var].shift(lag)
    y = df[TARGET_VAR]
    combined = pd.concat([df_lagged, y], axis=1).dropna()
    return combined.drop(columns=TARGET_VAR), combined[TARGET_VAR]

def compute_x_mse(y_pred, target_dates):
    if y_pred.empty:
        return np.nan
    x = np.arange(len(y_pred))
    poly = np.poly1d(np.polyfit(x, y_pred.values, 3))
    zeros = poly_zeros(poly)
    max_x, min_x = poly_extrema(poly)
    if len(zeros) < 2 or max_x is None or min_x is None:
        return np.nan
    pred_events = {
        'first_zero': zeros[0],
        'max': max_x,
        'second_zero': zeros[1],
        'min': min_x,
        'last_zero': zeros[-1] if len(zeros) >= 3 else zeros[1]
    }
    forecast_start = y_pred.index[0]
    target_x = {k: target_date_to_x(v, forecast_start) for k,v in target_dates.items()}
    return np.mean([(pred_events[k] - target_x[k])**2 for k in pred_events])

# === Generate valid variable sets ===
print("🧠 Generating valid variable sets...")
spearman_corr = df[all_vars].corr(method='spearman')
valid_var_sets = []

for n_vars in range(1, MAX_VARS+1):
    for var_combo in itertools.combinations(all_vars, n_vars):
        sub_corr = spearman_corr.loc[var_combo, var_combo]
        upper = sub_corr.where(np.triu(np.ones(sub_corr.shape), k=1).astype(bool))
        if (upper.abs() < CORR_THRESHOLD).all().all():
            valid_var_sets.append(['Difference'] + list(var_combo))

# === Brute force with printed progress ===
def evaluate_combo(var_list, lag_list):
    if len(var_list) * len(lag_list) > MAX_FEATURES:
        return None
    try:
        X_train, y_train = create_lagged_features(train, var_list, lag_list)
        X_test, y_test = create_lagged_features(combined, var_list, lag_list)
        X_test = X_test.loc[test.index.intersection(X_test.index)]
        y_test = y_test.loc[test.index.intersection(y_test.index)]
        X_train_const = sm.add_constant(X_train)
        X_test_const = sm.add_constant(X_test)
        model = sm.OLS(y_train, X_train_const).fit()
        y_pred = model.predict(X_test_const)
        x_mse = compute_x_mse(y_pred, TARGET_DATES)
        return {'x_mse': x_mse, 'variables': var_list, 'lags': lag_list}
    except:
        return None

lag_range = list(range(1, MAX_LAG+1))
top_models = []

for idx, var_set in enumerate(valid_var_sets):
    print(f"\n📊 Set {idx+1}/{len(valid_var_sets)} — {var_set}")
    max_lags = MAX_FEATURES // len(var_set)
    all_lag_combos = []
    for k in range(1, max_lags+1):
        lag_combos = list(itertools.combinations(lag_range, k))
        random.shuffle(lag_combos)
        all_lag_combos += lag_combos[:max(1, MAX_LAG_COMBOS // max_lags)]
    all_lag_combos = all_lag_combos[:MAX_LAG_COMBOS]

    print(f"🔁 Trying {len(all_lag_combos)} lag combinations...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(evaluate_combo)(var_set, list(lags)) for lags in all_lag_combos
    )
    valid = [r for r in results if r and not np.isnan(r['x_mse'])]
    print(f"✅ Found {len(valid)} valid models.")

    if valid:
        best = sorted(valid, key=lambda x: x['x_mse'])[:3]
        print("🏅 Top 3 in this set:")
        for i, model in enumerate(best, 1):
            print(f"  {i}. X-MSE={model['x_mse']:.4f}, Lags={model['lags']}")

    top_models.extend(valid)

# === Show top models
top_models = sorted(top_models, key=lambda x: x['x_mse'])[:10]
print("\n🏁 Final Top 10 Models:")
for i, model in enumerate(top_models, 1):
    print(f"Rank {i}: X-MSE={model['x_mse']:.4f}")
    print(f"    Variables: {model['variables']}")
    print(f"    Lags     : {model['lags']}\n")
