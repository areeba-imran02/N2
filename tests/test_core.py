import io, math, numpy as np, pandas as pd
from novexa import calculator as C, statistics_engine as S, ml_engine as M, document_engine as D, ai_router as R, artifact_generator as A

def raises(exc, f, *a):
    try: f(*a)
    except exc: return True
    except Exception: return False
    return False

def test_arith(): assert C.evaluate("2+3*4")[0] == 14 and C.evaluate("(2+3)*4")[0] == 20
def test_deg_rad():
    assert abs(C.evaluate("sin(30)", "DEG")[0] - 0.5) < 1e-12
    assert abs(C.evaluate("sin(pi/6)", "RAD")[0] - 0.5) < 1e-12
def test_degree_symbol_example():
    v, t = C.evaluate("sin(30°)+5^2", "RAD"); assert abs(v - 25.5) < 1e-12 and t[-1][0] == "result"
def test_errors():
    assert raises(C.CalcError, C.evaluate, "1/0") and raises(C.CalcError, C.evaluate, "2+")
    assert raises(C.CalcError, C.evaluate, "sqrt(-1)") and raises(C.CalcError, C.evaluate, "__import__('os')")
    assert raises(C.CalcError, C.evaluate, "fact(500)") and raises(C.CalcError, C.evaluate, "9**9**9")
def test_funcs():
    assert C.evaluate("fact(5)")[0] == 120 and abs(C.evaluate("ln(e)")[0] - 1) < 1e-12
    assert abs(C.evaluate("acos(0)", "DEG")[0] - 90) < 1e-9 and C.percent(200, 15) == 30
def test_describe():
    d = S.describe([1, 2, 2, 3, 4]); assert d["mean"] == 2.4 and d["median"] == 2 and d["mode"] == 2
    assert abs(d["std"] - np.std([1, 2, 2, 3, 4], ddof=1)) < 1e-12
def test_ttest():
    r = S.one_sample_t([5.1, 4.9, 5.2, 5.0, 5.3], 5.0)
    assert r["n"] == 5 and 0 <= r["p_value"] <= 1 and "h0" in r
def test_csv_validation():
    assert S.load_table(io.BytesIO(b"a,b\n1,2\n"), "x.csv").shape == (1, 2)
    assert raises(ValueError, S.load_table, io.BytesIO(b"x"), "x.txt")
def _clf():
    rng = np.random.RandomState(0); x = rng.randn(200, 3)
    return pd.DataFrame(x, columns=list("abc")).assign(y=np.where(x[:, 0] + .3 * rng.randn(200) > 0, "p", "n"))
def test_ml_classification():
    df = _clf(); r = M.train(df, "y", "Logistic Regression")
    assert not set(r["train_index"]) & set(r["test_index"]) and r["n_train"] + r["n_test"] == 200
    assert r["model"]["accuracy"] > r["baseline"]["accuracy"] and r["confusion_matrix"].sum() == r["n_test"]
    assert len(M.predict_rows(r, df.drop(columns="y").head(3))) == 3
def test_ml_regression():
    rng = np.random.RandomState(1); x = rng.rand(150); df = pd.DataFrame({"x": x, "y": 3 * x + .01 * rng.randn(150)})
    r = M.train(df, "y", "Linear Regression"); assert r["task"] == "regression" and r["model"]["r2"] > .95 and r["model"]["rmse"] < r["baseline"]["rmse"]
def test_ml_guards(): assert raises(ValueError, M.train, _clf().head(5), "y", "Decision Tree")
def test_kmeans():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(np.vstack([rng.randn(30, 2), rng.randn(30, 2) + 10]), columns=["a", "b"])
    l = M.kmeans(df, 2)["labels"]; assert len(set(l[:30])) == 1 and len(set(l[30:])) == 1 and l[0] != l[30]
def test_retrieval():
    ix = D.Index([{"page": 1, "text": "Photosynthesis converts light into chemical energy in chloroplasts."},
                  {"page": 2, "text": "The French revolution began in 1789 with the storming of the Bastille."}])
    assert ix.search("when did the French revolution begin")[0]["page"] == 2 and ix.search("quantum chromodynamics") == []
def test_router_missing_key(): assert raises(R.NotConfigured, R.Router("").generate, "hi")
class E(Exception):
    def __init__(s, code): s.code = code
def test_router_auth(): 
    def c(m, p): raise E(401)
    assert raises(R.AuthError, R.Router("k", call=c).generate, "hi")
def test_router_fallback_and_cooldown():
    calls = []
    def c(m, p):
        calls.append(m)
        if m == "a": raise E(429)
        return "ok"
    r = R.Router("k", ["a", "b"], call=c, sleep=lambda s: None)
    assert r.generate("x") == ("ok", "b") and r.generate("x")[1] == "b" and calls == ["a", "b", "b"]
def test_router_retry_then_all_down():
    n = []
    def c(m, p): n.append(m); raise E(503)
    r = R.Router("k", ["a", "b"], call=c, sleep=lambda s: None, retries=1)
    assert raises(R.AllModelsUnavailable, r.generate, "x") and n == ["a", "a", "b", "b"]
def test_router_invalid_not_retried():
    n = []
    def c(m, p): n.append(m); raise E(400)
    assert raises(R.AIError, R.Router("k", call=c).generate, "x") and len(n) == 1
def test_artifacts():
    df = pd.DataFrame({"a": [1, 2]})
    assert A.validate(A.xlsx_bytes(df), "xlsx") and A.validate(A.docx_report("T", [("h", "b")]), "docx") and A.validate(A.csv_bytes(df), "csv")
