"""Seven specialist agents + orchestrator. Status/duration come from real runs only."""
import time, json
from . import calculator as C, statistics_engine as S, ml_engine as M, coding as K
from .proof_graph import ProofGraph

STATES = ("Idle", "Planning", "Running", "Completed", "Failed", "Waiting for model", "Verification required")
class Agent:
    key = name = role = contract = ""
    def __init__(self): self.status, self.duration, self.error, self.runs, self.last = "Idle", None, None, 0, None
    def execute(self, payload, graph, parent):
        self.status, self.error, t0 = "Running", None, time.perf_counter()
        nid = graph.add("agent", f"{self.name} started", {"input_keys": sorted(payload)}, parent)
        try:
            out = self.run(payload, graph, nid); self.status = "Completed"
            graph.add("result", f"{self.name} finished", {"output": _brief(out)}, nid)
        except Exception as e:
            out, self.status, self.error = None, "Failed", f"{type(e).__name__}: {e}"
            graph.add("error", self.error[:60], {"error": self.error}, nid, "error")
        self.duration, self.runs, self.last = time.perf_counter() - t0, self.runs + 1, out
        return out
    def run(self, payload, graph, nid): raise NotImplementedError

def _brief(o):
    try: return json.loads(json.dumps(o, default=lambda x: str(x)[:200]))
    except Exception: return str(o)[:300]

class MathAgent(Agent):
    key, name, role = "math", "Mathematics Agent", "Safe evaluation, symbolic algebra, matrices"
    contract = "in: expr [+mode | symbolic_op,var | matrix_op,a,b] -> out: result, steps, trace"
    def run(self, p, g, nid):
        if p.get("matrix_op"):
            r = C.matrix_op(p["matrix_op"], p["a"], p.get("b")); g.add("tool", "sympy.Matrix", {"op": p["matrix_op"]}, nid)
            return {"result": r, "steps": [f"Parsed matrices", f"Applied {p['matrix_op']}"], "kind": "matrix"}
        if p.get("symbolic_op"):
            r, steps = C.symbolic(p["symbolic_op"], p["expr"], p.get("var", "x")); g.add("tool", "sympy", {"op": p["symbolic_op"]}, nid)
            return {"result": r, "steps": steps, "kind": "symbolic", "expr": p["expr"], "var": p.get("var", "x")}
        v, tr = C.evaluate(p["expr"], p.get("mode", "RAD")); g.add("tool", "calculator.evaluate", {"mode": p.get("mode", "RAD"), "events": len(tr)}, nid)
        return {"result": v, "display": C.fmt(v), "trace": tr, "steps": [f"{a}: {b}" for a, b in tr], "kind": "numeric", "expr": p["expr"], "mode": p.get("mode", "RAD")}

class StatsAgent(Agent):
    key, name, role = "stats", "Statistics Agent", "Descriptive stats, t-tests, regression"
    contract = "in: df, column [+mu0 | group_b | x,y] -> out: stats / test result"
    def run(self, p, g, nid):
        df, c = p["df"], p["column"]; out = {"describe": S.describe(df[c])}
        if "mu0" in p: out["test"] = S.one_sample_t(df[c], p["mu0"])
        if p.get("group_b"): out["test"] = S.welch_t(df[c], df[p["group_b"]])
        if p.get("y"): out["regression"] = S.linregress_df(df, c, p["y"])
        g.add("tool", "statistics_engine", {"column": c}, nid); out["column"] = c; return out

class MLAgent(Agent):
    key, name, role = "ml", "Machine Learning Agent", "Trains and evaluates scikit-learn models locally"
    contract = "in: df, target, model [, task] -> out: metrics, baseline, warnings"
    def run(self, p, g, nid):
        r = M.train(p["df"], p["target"], p["model"], p.get("task"))
        g.add("tool", "ml_engine.train", {"model": p["model"], "n_train": r["n_train"], "n_test": r["n_test"]}, nid); return r

class ResearchAgent(Agent):
    key, name, role = "research", "Research Agent", "Page-aware retrieval from uploaded documents"
    contract = "in: index, question -> out: passages with page numbers (empty = no evidence)"
    def run(self, p, g, nid):
        hits = p["index"].search(p["question"]); g.add("retrieval", f"{len(hits)} passage(s)", {"pages": [h["page"] for h in hits]}, nid)
        return {"hits": hits, "evidence_found": bool(hits)}

class CodingAgent(Agent):
    key, name, role = "coding", "Coding Assistant Agent", "Static analysis (no code execution)"
    contract = "in: code -> out: syntax result, issues, structure"
    def run(self, p, g, nid):
        r = K.analyze(p["code"]); g.add("tool", "coding.analyze (ast, not executed)", {"issues": len(r["issues"])}, nid); return r

class VerificationAgent(Agent):
    key, name, role = "verify", "Verification Agent", "Independent recomputation and evidence checks"
    contract = "in: kind + producing agent's output -> out: verified | partial | unverified + checks"
    def run(self, p, g, nid):
        o, checks = p["output"], []
        k = o.get("kind") if isinstance(o, dict) else None
        if k == "numeric":
            import sympy as sp
            ex = o["expr"].replace("^", "**").replace("°", "*pi/180")
            if o["mode"] == "DEG" and any(f in ex for f in ("sin", "cos", "tan", "asin", "acos", "atan")): checks.append(("sympy recompute", None, "skipped: DEG-mode trig not independently re-derived"))
            else:
                try:
                    ref = float(sp.N(sp.sympify(ex, locals={"ln": sp.log, "log": lambda v: sp.log(v, 10), "fact": sp.factorial, "e": sp.E})))
                    checks.append(("sympy recompute", abs(ref - float(o["result"])) <= 1e-9 * max(1, abs(ref)), f"reference={ref}"))
                except Exception as e: checks.append(("sympy recompute", None, f"could not recompute: {type(e).__name__}"))
        elif k == "symbolic" and o.get("expr"):
            checks.append(("syntactic", None, "symbolic output not independently re-derived"))
        elif "model" in o and "train_index" in o:
            checks.append(("train/test disjoint", not set(o["train_index"]) & set(o["test_index"]), ""))
            vals = [v for v in o["model"].values()]; checks.append(("metrics finite", all(v == v and abs(v) < 1e12 for v in vals), ""))
        elif "describe" in o:
            import pandas as pd, numpy as np
            col = p["df"][o["column"]].dropna(); checks.append(("mean recompute", abs(col.mean() - o["describe"]["mean"]) < 1e-9, ""))
        elif "hits" in o:
            ok = all(any(pg["page"] == h["page"] and h["text"][:80] in pg["text"] for pg in p["index"].pages) for h in o["hits"])
            checks.append(("cited text exists on cited page", ok if o["hits"] else None, "no evidence retrieved" if not o["hits"] else ""))
        else: checks.append(("no applicable check", None, ""))
        checks = [(n, None if ok is None else bool(ok), d) for n, ok, d in checks]
        res = [c[1] for c in checks]
        verdict = "unverified" if False in res else ("verified" if res and all(r is True for r in res) else "partial")
        for n, ok, d in checks: g.add("verification", f"{n}: {ok}", {"detail": d}, nid, "verified" if ok else ("error" if ok is False else "warning"))
        return {"verdict": verdict, "checks": checks}

class Orchestrator(Agent):
    key, name, role = "orchestrator", "NOVEXA Orchestrator", "Classifies requests, dispatches specialists, tracks results"
    contract = "in: request + payload -> out: routing decision, agent output, verification"
    RULES = [("coding", ("code", "python", "debug", "function")), ("ml", ("train", "model", "classif", "regress", "predict")),
             ("stats", ("mean", "t-test", "median", "correlat", "statistic")), ("research", ("pdf", "document", "page", "paper")),
             ("math", ("solve", "derivative", "integral", "matrix", "calculate", "+", "sin", "sqrt"))]
    def __init__(self):
        super().__init__(); self.agents = {a.key: a for a in (MathAgent(), StatsAgent(), MLAgent(), ResearchAgent(), CodingAgent(), VerificationAgent())}
        self.graph = ProofGraph()
    def route(self, request, hint=None):
        if hint in self.agents: return hint, f"explicit selection by caller ({hint})"
        low = request.lower()
        for k, kws in self.RULES:
            m = [w for w in kws if w in low]
            if m: return k, f"keyword match: {m}"
        return None, "no keyword matched"
    def handle(self, request, payload, hint=None):
        self.status = "Planning"; t0 = time.perf_counter()
        root = self.graph.add("request", request[:60], {"request": request})
        key, why = self.route(request, hint)
        self.graph.add("decision", f"route -> {key}", {"reason": why}, root)
        if key is None:
            self.status, self.error = "Failed", "could not classify request"; self.graph.add("error", self.error, {}, root, "error"); return {"agent": None, "reason": why, "output": None}
        self.status = "Running"; ag = self.agents[key]; out = ag.execute(payload, self.graph, root); ver = None
        if out is not None and key != "coding":
            v = self.agents["verify"]; ver = v.execute({"output": out, **{k: payload[k] for k in ("df", "index") if k in payload}}, self.graph, root)
        self.status = "Completed" if out is not None else "Failed"; self.duration = time.perf_counter() - t0; self.runs += 1
        self.graph.add("conclusion", "task " + self.status.lower(), {"agent": key, "verdict": ver and ver["verdict"]}, root, "ok" if out is not None else "error")
        return {"agent": key, "reason": why, "output": out, "verification": ver, "error": ag.error}
