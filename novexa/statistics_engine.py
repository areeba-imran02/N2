import numpy as np, pandas as pd
from scipy import stats

def describe(s):
    s = pd.to_numeric(pd.Series(s), errors="coerce").dropna()
    if s.empty: raise ValueError("no numeric values")
    nan = float("nan")
    return {"n": int(s.size), "mean": float(s.mean()), "median": float(s.median()), "mode": float(s.mode().iloc[0]),
            "variance": float(s.var(ddof=1)) if s.size > 1 else nan, "std": float(s.std(ddof=1)) if s.size > 1 else nan,
            "q1": float(s.quantile(.25)), "q3": float(s.quantile(.75))}

def one_sample_t(x, mu0=0.0, alpha=0.05):
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), float)
    if x.size < 2: raise ValueError("need at least 2 observations")
    r = stats.ttest_1samp(x, mu0)
    ci = stats.t.interval(1 - alpha, x.size - 1, loc=x.mean(), scale=stats.sem(x))
    return {"test": "One-sample t-test", "h0": f"population mean = {mu0}", "n": int(x.size),
            "statistic": float(r.statistic), "p_value": float(r.pvalue), "alpha": alpha,
            "reject_h0": bool(r.pvalue < alpha), "ci": (float(ci[0]), float(ci[1])),
            "assumptions": "Roughly normal data or large n; independent observations. Significance is not practical importance or causation."}

def welch_t(a, b, alpha=0.05):
    a, b = (np.asarray(pd.Series(v).dropna(), float) for v in (a, b))
    if a.size < 2 or b.size < 2: raise ValueError("need at least 2 observations per group")
    r = stats.ttest_ind(a, b, equal_var=False)
    return {"test": "Welch two-sample t-test", "h0": "group means are equal", "n": (int(a.size), int(b.size)),
            "statistic": float(r.statistic), "p_value": float(r.pvalue), "alpha": alpha, "reject_h0": bool(r.pvalue < alpha),
            "assumptions": "Independent groups; roughly normal or large samples."}

def linreg(x, y):
    r = stats.linregress(x, y)
    return {"slope": r.slope, "intercept": r.intercept, "r2": r.rvalue ** 2, "p_value": r.pvalue, "n": len(x)}

def load_table(file, name, max_mb=10, max_rows=50000, max_cols=100):
    file.seek(0, 2); size = file.tell(); file.seek(0)
    if size > max_mb * 1024 * 1024: raise ValueError(f"file exceeds {max_mb} MB")
    n = name.lower()
    if n.endswith(".csv"): df = pd.read_csv(file)
    elif n.endswith(".xlsx"): df = pd.read_excel(file, engine="openpyxl")
    else: raise ValueError("only CSV and XLSX are supported")
    if df.empty: raise ValueError("dataset is empty")
    if df.shape[0] > max_rows or df.shape[1] > max_cols: raise ValueError("dataset too large")
    return df

def quality_report(df):
    return {"rows": len(df), "cols": df.shape[1], "duplicates": int(df.duplicated().sum()),
            "missing": df.isna().sum()[lambda s: s > 0].to_dict(), "dtypes": df.dtypes.astype(str).to_dict()}

def linregress_df(df, x, y):
    d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3: raise ValueError("need at least 3 complete rows")
    return linreg(d[x].values, d[y].values)
