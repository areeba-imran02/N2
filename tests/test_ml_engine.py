import pandas as pd
from novexa.ml_engine import train_supervised, cluster_data
def test_classification():
    df=pd.DataFrame({"x":list(range(30)),"z":[i%3 for i in range(30)],"y":["a" if i%2 else "b" for i in range(30)]})
    r=train_supervised(df,"y","classification","Decision Tree")
    assert 0 <= r["metrics"]["accuracy"] <= 1 and r["n_test"]>0
def test_regression():
    df=pd.DataFrame({"x":list(range(30)),"y":[2*i+1 for i in range(30)]})
    r=train_supervised(df,"y","regression","Linear Regression")
    assert r["metrics"]["mae"] < 1e-8
def test_cluster():
    df=pd.DataFrame({"x":[0,1,10,11,20,21],"y":[0,1,10,11,20,21]})
    out,model=cluster_data(df,["x","y"],3)
    assert "Cluster" in out and len(out)==6
