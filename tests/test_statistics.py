import pandas as pd
from novexa.statistics_engine import numeric_summary, one_sample_ttest
def test_summary():
    r=numeric_summary(pd.Series([1,2,3]))
    assert r["mean"]==2 and r["median"]==2 and r["n"]==3
def test_ttest():
    r=one_sample_ttest(pd.Series([1,2,3,4]),0)
    assert r["n"]==4 and 0<=r["p_value"]<=1
