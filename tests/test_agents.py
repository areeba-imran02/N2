import numpy as np, pandas as pd
from novexa import agents as AG, calculator as C, coding as K, document_engine as D, artifact_generator as A
from novexa.proof_graph import ProofGraph
def raises(exc, f, *a):
    try: f(*a)
    except exc: return True
    except Exception: return False
    return False
def test_symbolic_and_matrix():
    assert C.symbolic("diff", "x^2")[0] == "2*x" and C.symbolic("solve", "x^2=4")[0] in ("[-2, 2]",)
    assert C.matrix_op("det", "[[1,2],[3,4]]") == "-2" and raises(C.CalcError, C.matrix_op, "inv", "[[1,2],[2,4]]")
def test_symbolic_injection():
    for bad in ("__import__('os').system('x')", "().__class__", "open('f')", "lambda: 1"): assert raises(C.CalcError, C.symbolic, "simplify", bad)
def test_orchestrator_math_verified():
    o = AG.Orchestrator(); r = o.handle("calculate 2+3*4", {"expr": "2+3*4", "mode": "RAD"})
    assert r["agent"] == "math" and r["output"]["result"] == 14 and r["verification"]["verdict"] == "verified"
    assert o.agents["math"].status == "Completed" and o.agents["math"].duration > 0
def test_failure_state_recorded():
    o = AG.Orchestrator(); r = o.handle("calculate 1/0", {"expr": "1/0"})
    assert o.agents["math"].status == "Failed" and r["output"] is None and any(n["kind"] == "error" for n in o.graph.nodes)
def test_unroutable():
    o = AG.Orchestrator(); assert o.handle("hello there", {})["agent"] is None and o.status == "Failed"
def test_graph_edges_reference_nodes():
    o = AG.Orchestrator(); o.handle("calculate 1+1", {"expr": "1+1"}); ids = {n["id"] for n in o.graph.nodes}
    assert all(a in ids and b in ids for a, b in o.graph.edges) and "digraph" in o.graph.to_dot()
def test_ml_agent_and_verification():
    rng = np.random.RandomState(0); x = rng.randn(120, 2); df = pd.DataFrame(x, columns=["a", "b"]).assign(y=np.where(x[:, 0] > 0, "p", "n"))
    r = AG.Orchestrator().handle("train model", {"df": df, "target": "y", "model": "Decision Tree"})
    assert r["agent"] == "ml" and r["verification"]["verdict"] == "verified"
def test_research_no_evidence():
    ix = D.Index([{"page": 3, "text": "Cats are small mammals that purr."}]); o = AG.Orchestrator()
    r = o.handle("pdf question", {"index": ix, "question": "quantum field theory"}, hint="research")
    assert r["output"]["hits"] == [] and r["verification"]["verdict"] == "partial"
    r2 = o.handle("pdf question", {"index": ix, "question": "do cats purr"}, hint="research")
    assert r2["output"]["hits"][0]["page"] == 3 and r2["verification"]["verdict"] == "verified"
def test_stats_agent():
    df = pd.DataFrame({"v": [1., 2, 3, 4, 5.]}); r = AG.Orchestrator().handle("mean", {"df": df, "column": "v", "mu0": 3.0})
    assert r["output"]["describe"]["mean"] == 3.0 and r["verification"]["verdict"] == "verified"
def test_coding_static():
    r = K.analyze("def f(x=[]):\n    try:\n        eval(x)\n    except:\n        pass\n"); assert len(r["issues"]) >= 3
    assert K.analyze("def (:")["ok"] is False
def test_summary_terms():
    pg = [{"page": 1, "text": "Neural networks learn representations from data. Training adjusts weights using gradients. Evaluation uses held out data."}]
    s = D.extractive_summary(pg, 2); assert s and all(p == 1 for p, _ in s) and D.key_terms(pg)
    assert raises(ValueError, D.validate_upload, b"abc", "x.pdf")
def test_report_make():
    assert A.validate(A.make("docx", "T", [("a", "b")]), "docx") and raises(ValueError, A.make, "txt", "T", [])
