import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)

CLS = {"Logistic Regression": lambda: LogisticRegression(max_iter=1000),
       "Decision Tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
       "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, n_jobs=1)}
REG = {"Linear Regression": LinearRegression,
       "Decision Tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
       "Random Forest": lambda: RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0, n_jobs=1)}

def infer_task(y):
    if y.dtype == object or str(y.dtype) in ("category", "bool") or (y.nunique() <= 10 and y.nunique() / max(len(y), 1) < .05):
        return "classification"
    return "regression"

def _pre(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])

def train(df, target, model_name, task=None, test_size=0.25, seed=42):
    if target not in df.columns: raise ValueError("target column not found")
    df = df.dropna(subset=[target])
    if len(df) < 20: raise ValueError("need at least 20 rows with a target value")
    X, y = df.drop(columns=[target]), df[target]
    if X.shape[1] == 0: raise ValueError("no feature columns")
    task = task or infer_task(y)
    warnings, strat = [], None
    if task == "classification":
        vc = y.value_counts()
        if len(vc) < 2: raise ValueError("target has only one class")
        if vc.min() >= 2: strat = y
        else: warnings.append("a class has <2 samples; split is not stratified")
        if vc.min() / vc.max() < 0.2: warnings.append("class imbalance: prefer F1 over accuracy")
    if len(df) < 100: warnings.append("small dataset: metrics are high-variance")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    zoo = CLS if task == "classification" else REG
    if model_name not in zoo: raise ValueError(f"unknown model {model_name}")
    pipe = Pipeline([("pre", _pre(Xtr)), ("m", zoo[model_name]())]).fit(Xtr, ytr)  # preprocessing fit on train only
    dummy = DummyClassifier(strategy="most_frequent") if task == "classification" else DummyRegressor()
    base = Pipeline([("pre", _pre(Xtr)), ("m", dummy)]).fit(Xtr, ytr)
    res = {"task": task, "model_name": model_name, "n_train": len(Xtr), "n_test": len(Xte),
           "train_index": list(Xtr.index), "test_index": list(Xte.index), "warnings": warnings, "pipeline": pipe}
    for tag, p in (("model", pipe), ("baseline", base)):
        pr = p.predict(Xte)
        if task == "classification":
            res[tag] = {"accuracy": accuracy_score(yte, pr), "f1_weighted": f1_score(yte, pr, average="weighted", zero_division=0)}
            if tag == "model":
                labels = sorted(y.unique(), key=str)
                res.update(labels=labels, confusion_matrix=confusion_matrix(yte, pr, labels=labels),
                           report=classification_report(yte, pr, zero_division=0))
        else:
            res[tag] = {"mae": mean_absolute_error(yte, pr), "rmse": float(np.sqrt(mean_squared_error(yte, pr))), "r2": r2_score(yte, pr)}
    return res

def predict_rows(result, rows): return result["pipeline"].predict(rows)

def kmeans(df, k=3, seed=0):
    X = df.select_dtypes(include="number").dropna()
    if X.shape[1] < 2 or len(X) < k + 1: raise ValueError("need >=2 numeric columns and more rows than clusters")
    Z = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    return {"labels": km.labels_, "inertia": km.inertia_, "pca": PCA(2, random_state=seed).fit_transform(Z), "index": X.index,
            "note": "Clusters are geometric groupings, not validated categories."}

def anomalies(df, contamination=0.05, seed=0):
    X = df.select_dtypes(include="number").dropna()
    if len(X) < 20: raise ValueError("need at least 20 rows")
    flags = IsolationForest(contamination=contamination, random_state=seed).fit_predict(X) == -1
    return {"flags": flags, "index": X.index, "note": "Statistical outliers only; not evidence of fraud or any condition."}
