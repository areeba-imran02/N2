import json, pathlib
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px
from novexa import calculator as C, statistics_engine as S, ml_engine as M, document_engine as D
from novexa import ai_router as R, artifact_generator as A, coding as K, agents as AG

ROOT = pathlib.Path(__file__).parent
st.set_page_config(page_title="NOVEXA AI", page_icon=str(ROOT / "assets/logo.svg"), layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(160deg,#14141c 0%,#1a1630 100%)}
h1,h2,h3{letter-spacing:-.01em} [data-testid=stSidebar]{background:#171722;border-right:1px solid #2c2845}
.card{border:1px solid #2c2845;border-radius:14px;padding:.8rem 1rem;background:#1d1d2b;margin-bottom:.6rem}
.pill{display:inline-block;padding:1px 10px;border-radius:99px;border:1px solid #8b7cf6;color:#cfc8ff;font-size:.78rem}
</style>""", unsafe_allow_html=True)
ss = st.session_state
for k, v in {"chat": [], "tasks": [], "trace": [], "hist": [], "expr": "sin(30°) + 5^2", "doc_qa": [], "report": {}}.items(): ss.setdefault(k, v)
if "orch" not in ss: ss.orch = AG.Orchestrator()
orch = ss.orch

def secret(name, default=""):
    try: return st.secrets.get(name, default)
    except Exception: return default
def router():
    if "router" not in ss:
        try: models = json.loads(secret("GEMINI_MODELS", "") or "null")
        except Exception: models = None
        ss.router = R.Router(secret("GEMINI_API_KEY"), models)
    return ss.router
def ask(prompt, node_parent=None):
    rt = router()
    try:
        with st.spinner("Waiting for Gemini…"): out, model = rt.generate(prompt)
        orch.graph.add("model", f"Gemini: {model}", {"attempts": rt.status["attempts"]}, node_parent); return out
    except R.AIError as e:
        orch.graph.add("error", f"model error: {type(e).__name__}", {"attempts": rt.status["attempts"], "msg": str(e)}, node_parent, "error")
        st.error(str(e)); return None
def run_task(request, payload, hint):
    r = orch.handle(request, payload, hint); a = orch.agents.get(r["agent"])
    ss.tasks.append({"request": request[:80], "agent": r["agent"], "status": "Completed" if r["output"] is not None else "Failed",
                     "verdict": (r["verification"] or {}).get("verdict", "n/a"), "seconds": round(a.duration, 4) if a and a.duration else None})
    ss.tasks = ss.tasks[-100:]; return r
def verdict_badge(r):
    v = (r.get("verification") or {}).get("verdict")
    if v: st.markdown(f"<span class='pill'>verification: {v}</span>", unsafe_allow_html=True)
def img(name): return str(ROOT / "assets/agents" / name)
def dl_row(title, sections, key):
    if not any(str(b).strip() for _, b in sections): return
    c = st.columns(4)
    for col, kind in zip(c, ("txt", "docx", "pdf")):
        try: col.download_button(f"Report .{kind}", A.make(kind, title, sections), f"{key}.{kind}", key=f"{key}{kind}")
        except Exception as e: col.caption(f".{kind} unavailable: {str(e)[:50]}")

PAGES = ["Overview", "AI Problem Solver", "Research & PDF Lab", "Scientific Calculator", "Machine Learning Studio",
         "Statistics & Data Lab", "Python & Coding Lab", "Agent Control Center", "Task History", "Settings & AI Connection"]
with st.sidebar:
    st.image(str(ROOT / "assets/logo.svg"), width=56); st.title("NOVEXA AI"); st.caption("Beyond Answers. Into Action.")
    page = st.radio("Workspace", PAGES, key="page")
main, side = st.columns([3, 1.35], gap="large")

with main:
    if page == "Overview":
        st.header("Problem-solving workspace")
        st.write("Seven specialist agents run real local engines (calculator, scikit-learn, SciPy, TF-IDF retrieval). Gemini is used for explanations and chat only. Every task is recorded in the Proof Graph under Agent Control Center.")
        a, b, c = st.columns(3); a.metric("Tasks this session", len(ss.tasks)); b.metric("Proof Graph events", len(orch.graph.nodes)); c.metric("Gemini key", "configured" if router().key else "missing")
    elif page == "AI Problem Solver":
        st.header("AI Problem Solver"); st.caption("The Orchestrator classifies your request and picks a specialist; the reason is recorded.")
        req = st.text_input("Describe the task", "calculate sqrt(144) + 3^2")
        if st.button("Solve", type="primary"):
            low = req.lower()
            payload = {"expr": req.lower().replace("calculate", "").strip(), "mode": "RAD"}
            if orch.route(req)[0] in ("ml", "stats", "research"): st.info("This needs data or a document: use the ML, Statistics or Research workspace.")
            else:
                r = run_task(req, payload, None); st.write("Routed to:", r["agent"], "—", r["reason"])
                if r["output"] is None: st.error(r.get("error") or "Task failed: " + r["reason"])
                else: st.success(r["output"].get("display", str(r["output"])[:300])); verdict_badge(r); ss.report = {"Request": req, "Result": r["output"].get("display", "")}
    elif page == "Research & PDF Lab":
        st.header("Research & PDF Lab")
        up = st.file_uploader("PDF or DOCX (max 15 MB, 300 pages)", type=["pdf", "docx"])
        if up and ss.get("doc_name") != up.name:
            try:
                data = up.getvalue(); D.validate_upload(data, up.name)
                pages = D.extract_pdf_pages(data) if up.name.lower().endswith(".pdf") else D.extract_docx(data)
                ss.pages, ss.index, ss.doc_name, ss.doc_qa = pages, D.Index(pages), up.name, []
            except Exception as e: st.error(str(e))
        if ss.get("index"):
            pg = ss.pages; st.caption(f"{ss.doc_name}: {len(pg)} pages, {sum(1 for p in pg if p['text'])} with text")
            t1, t2, t3, t4 = st.tabs(["Ask", "Summary & concepts", "Quiz", "Text similarity"])
            with t1:
                q = st.text_input("Question")
                if q and st.button("Search & answer", type="primary"):
                    r = run_task("pdf question: " + q, {"index": ss.index, "question": q}, "research")
                    if not r["output"] or not r["output"]["hits"]: st.warning("No supporting passage found in this document. No answer will be invented.")
                    else:
                        hits = r["output"]["hits"]; verdict_badge(r)
                        for h in hits:
                            with st.expander(f"Source — page {h['page']} (similarity {h['score']:.2f})", expanded=True): st.write(h["text"])
                        ctx = "\n\n".join(f"[p.{h['page']}] {h['text']}" for h in hits); prev = "\n".join(f"Q:{a}\nA:{b}" for a, b in ss.doc_qa[-3:])
                        ans = ask(f"Answer ONLY from these excerpts and cite pages. If insufficient, say so.\nEarlier turns:\n{prev}\n\nExcerpts:\n{ctx}\n\nQuestion: {q}")
                        if ans: st.markdown("**Model interpretation (not source text):**\n\n" + ans); ss.doc_qa.append((q, ans))
            with t2:
                try:
                    st.write("**Key terms (TF-IDF):**", ", ".join(D.key_terms(pg)))
                    st.write("**Extractive summary** (real sentences, page-tagged):")
                    for p, s in D.extractive_summary(pg): st.write(f"- {s} *(p.{p})*")
                except ValueError as e: st.warning(str(e))
            with t3:
                if st.button("Generate quiz with Gemini"):
                    base = "\n".join(f"[p.{p}] {s}" for p, s in D.extractive_summary(pg, 8))
                    qz = ask(f"Write 5 quiz questions with answers using ONLY this text; cite page numbers:\n{base}")
                    if qz: st.markdown(qz)
            with t4:
                tx = [p for p in pg if p["text"]][:12]
                if len(tx) >= 2:
                    m = D.similarity_matrix([p["text"] for p in tx]); lab = [f"p.{p['page']}" for p in tx]
                    st.plotly_chart(px.imshow(m, x=lab, y=lab, color_continuous_scale="Purples"), use_container_width=True); st.caption("Similarity is lexical overlap, not factual agreement.")
                else: st.info("Need at least 2 pages with text.")
        else: st.info("Upload a text-based PDF or DOCX. Scanned PDFs are rejected because OCR is not available.")
    elif page == "Scientific Calculator":
        st.header("Scientific Calculator")
        mode = st.radio("Angle mode", ["DEG", "RAD"], horizontal=True)
        def push(t): ss.expr = ss.expr + t
        def back(): ss.expr = ss.expr[:-1]
        def clear(): ss.expr = ""
        rows = [["7", "8", "9", "/", "sin(", "asin("], ["4", "5", "6", "*", "cos(", "acos("], ["1", "2", "3", "-", "tan(", "atan("],
                ["0", ".", "(", ")", "ln(", "log("], ["^", "pi", "e", "+", "sqrt(", "fact("], ["abs(", "%", "°", "exp(", "cbrt(", "**"]]
        for i, row in enumerate(rows):
            for col, t in zip(st.columns(len(row)), row): col.button(t, key=f"k{i}{t}", on_click=push, args=(t if t != "%" else "/100",), use_container_width=True)
        c1, c2 = st.columns(2); c1.button("⌫", on_click=back, use_container_width=True); c2.button("Clear", on_click=clear, use_container_width=True)
        st.text_input("Expression", key="expr")
        if st.button("Calculate", type="primary"):
            r = run_task("calculate " + ss.expr, {"expr": ss.expr, "mode": mode}, "math")
            if r["output"] is None: st.error(f"Cannot compute: {r['error']}")
            else:
                o = r["output"]; st.metric("Result", o["display"]); st.code(o["display"]); verdict_badge(r); ss.trace = o["trace"][-50:]
                ss.hist = ([(ss.expr, mode, o["display"])] + ss.hist)[:50]; ss.report = {"Expression": ss.expr, "Mode": mode, "Result": o["display"], "Trace": "\n".join(o["steps"])}
        with st.expander("Execution trace (events that actually ran)", expanded=True): st.write(ss.trace or "No calculation yet.")
        with st.expander("History"): st.table(pd.DataFrame(ss.hist, columns=["Expression", "Mode", "Result"])) if ss.hist else st.caption("Empty")
        t1, t2 = st.tabs(["Symbolic", "Matrix"])
        with t1:
            sop = st.selectbox("Operation", ["simplify", "diff", "integrate", "solve"]); sx = st.text_input("Expression (use = for solve)", "x^2 - 4 = 0")
            if st.button("Run symbolic"):
                r = run_task("solve symbolic " + sx, {"expr": sx, "symbolic_op": sop, "var": "x"}, "math")
                if r["output"] is None: st.error(r["error"])
                else: st.success(r["output"]["result"]); st.write(r["output"]["steps"]); verdict_badge(r)
        with t2:
            mop = st.selectbox("Matrix op", ["det", "inv", "transpose", "eigvals", "mul", "add"]); ma = st.text_input("A", "[[1,2],[3,4]]"); mb = st.text_input("B (mul/add)", "[[0,1],[1,0]]")
            if st.button("Run matrix"):
                r = run_task("matrix " + mop, {"matrix_op": mop, "a": ma, "b": mb}, "math")
                st.error(r["error"]) if r["output"] is None else (st.success(r["output"]["result"]), st.write(r["output"]["steps"]))
        dl_row("NOVEXA calculation worksheet", list(ss.report.items()), "calc")
    elif page in ("Machine Learning Studio", "Statistics & Data Lab"):
        st.header(page); up = st.file_uploader("CSV or XLSX (max 10 MB)", type=["csv", "xlsx"])
        demo = st.checkbox("Use built-in demo dataset (scikit-learn Iris — NOT your data)")
        df = None
        try:
            if up: df = S.load_table(up, up.name); src = up.name
            elif demo:
                from sklearn.datasets import load_iris; df = load_iris(as_frame=True).frame; src = "Iris demo dataset"
        except Exception as e: st.error(str(e))
        if df is None: st.info("Upload a dataset or tick the demo option.")
        else:
            ss.ds_name = src; st.caption(f"Source: {src}"); st.dataframe(df.head(50), use_container_width=True); st.json(S.quality_report(df))
            num = df.select_dtypes("number").columns.tolist()
            if page == "Statistics & Data Lab":
                if not num: st.warning("No numeric columns.")
                else:
                    c = st.selectbox("Column", num); t1, t2, t3, t4 = st.tabs(["Describe & tests", "Distribution", "Correlation", "Regression"])
                    with t1:
                        mu = st.number_input("H0 mean (one-sample t)", value=0.0); other = st.selectbox("Welch vs column", [None] + [n for n in num if n != c])
                        if st.button("Run analysis", type="primary"):
                            p = {"df": df, "column": c, "mu0": mu}; 
                            if other: p["group_b"] = other
                            r = run_task("mean t-test " + c, p, "stats")
                            if r["output"] is None: st.error(r["error"])
                            else:
                                o = r["output"]; st.json(o["describe"]); st.json(o["test"]); verdict_badge(r)
                                ss.report = {"Dataset": src, "Descriptive": json.dumps(o["describe"], indent=1), "Test": json.dumps(o["test"], indent=1, default=str)}
                    with t2: st.plotly_chart(px.histogram(df, x=c, marginal="box"), use_container_width=True)
                    with t3:
                        if len(num) > 1: cm = df[num].corr(); st.plotly_chart(px.imshow(cm, text_auto=".2f", color_continuous_scale="Purples"), use_container_width=True); st.download_button("Correlation CSV", cm.to_csv().encode(), "corr.csv")
                        st.caption("Correlation does not imply causation.")
                    with t4:
                        y = st.selectbox("Y column", [n for n in num if n != c] or [None])
                        if y and st.button("Fit regression"):
                            r = run_task("statistic regression", {"df": df, "column": c, "y": y}, "stats"); 
                            if r["output"] is None: st.error(r["error"])
                            else: st.json(r["output"]["regression"]); st.plotly_chart(px.scatter(df, x=c, y=y, trendline="ols"), use_container_width=True)
                    dl_row("NOVEXA statistics report", list(ss.report.items()), "stats")
            else:
                t1, t2, t3 = st.tabs(["Supervised", "Clustering (K-Means + PCA)", "Anomalies (Isolation Forest)"])
                with t1:
                    t = st.selectbox("Target", df.columns, index=len(df.columns) - 1); inf = M.infer_task(df[t])
                    task = st.radio("Task", ["classification", "regression"], index=["classification", "regression"].index(inf), horizontal=True); st.caption(f"Inferred: {inf}")
                    name = st.selectbox("Model", list(M.CLS if task == "classification" else M.REG))
                    if st.button("Train & evaluate", type="primary"):
                        r = run_task("train model", {"df": df, "target": t, "model": name, "task": task}, "ml")
                        if r["output"] is None: st.error(r["error"])
                        else: ss.ml, ss.ml_df, ss.ml_t = r["output"], df, t; verdict_badge(r)
                    o = ss.get("ml")
                    if o:
                        for w in o["warnings"]: st.warning(w)
                        a, b = st.columns(2); a.write("**Model (held-out test set)**"); a.json(o["model"]); b.write("**Baseline**"); b.json(o["baseline"])
                        if o["task"] == "classification":
                            lab = [str(x) for x in o["labels"]]; st.plotly_chart(px.imshow(o["confusion_matrix"], x=lab, y=lab, text_auto=True, labels=dict(x="Predicted", y="Actual")), use_container_width=True); st.code(o["report"])
                        te = ss.ml_df.loc[o["test_index"]]; pred = pd.DataFrame({"actual": te[ss.ml_t], "predicted": o["pipeline"].predict(te.drop(columns=ss.ml_t))}, index=te.index)
                        st.download_button("Test-set predictions CSV", pred.to_csv().encode(), "predictions.csv")
                        if st.button("Explain with Gemini"):
                            e = ask("Explain these REAL scikit-learn results to a beginner. Do not invent numbers.\n" + json.dumps({k: o[k] for k in ("task", "model_name", "model", "baseline", "n_train", "n_test", "warnings")}, default=str))
                            if e: st.markdown(e)
                        dl_row("NOVEXA ML evaluation", [("Model", o["model_name"]), ("Metrics", json.dumps(o["model"], default=str)), ("Baseline", json.dumps(o["baseline"], default=str)), ("Warnings", "\n".join(o["warnings"]) or "none")], "ml")
                with t2:
                    k = st.slider("Clusters", 2, 8, 3)
                    if st.button("Run K-Means"):
                        try: res = M.kmeans(df, k); st.plotly_chart(px.scatter(x=res["pca"][:, 0], y=res["pca"][:, 1], color=res["labels"].astype(str), labels=dict(x="PC1", y="PC2")), use_container_width=True); st.caption(f"inertia={res['inertia']:.2f}. {res['note']}"); st.download_button("Cluster labels CSV", pd.DataFrame({"cluster": res["labels"]}, index=res["index"]).to_csv().encode(), "clusters.csv")
                        except Exception as e: st.error(str(e))
                with t3:
                    cont = st.slider("Expected outlier share", 0.01, 0.2, 0.05)
                    if st.button("Detect anomalies"):
                        try:
                            res = M.anomalies(df, cont); d = df.loc[res["index"]].assign(anomaly=res["flags"]); st.write(f"{int(res['flags'].sum())} flagged"); st.caption(res["note"]); st.dataframe(d[d.anomaly]); st.download_button("Anomaly flags CSV", d.to_csv().encode(), "anomalies.csv")
                        except Exception as e: st.error(str(e))
    elif page == "Python & Coding Lab":
        st.header("Python & Coding Lab"); st.warning("Code execution is DISABLED: no isolated sandbox is available. Your code is analysed statically and never run.")
        st.selectbox("Language", ["Python"]); code = st.text_area("Code", "def add(a, b=[]):\n    return a + b\n", height=220); st.code(code, language="python")
        if st.button("Static analysis", type="primary"):
            r = run_task("python code review", {"code": code}, "coding")
            if r["output"] is None: st.error(r["error"])
            elif not r["output"]["ok"]: st.error("Syntax error: " + r["output"]["syntax_error"])
            else: st.json(r["output"]["stats"]); [st.warning(i) for i in r["output"]["issues"]] or None; st.success("No issues found") if not r["output"]["issues"] else None
        mode = st.selectbox("Gemini assistance", ["Explain", "Review", "Debug", "Suggest tests"])
        if st.button("Ask Gemini"):
            a = ask(f"{mode} this Python code. Do not claim you ran it.\n```python\n{code[:6000]}\n```")
            if a: st.markdown(a)
    elif page == "Agent Control Center":
        st.header("Agent Control Center"); st.caption("Status and timings below come from actual runs this session.")
        info = {"orchestrator": "orchestrator.svg", "research": "research.svg", "math": "mathematics.svg", "stats": "statistics.svg", "ml": "ml_engineer.svg", "coding": "coding.svg", "verify": "verification.svg"}
        for k, f in info.items():
            ag = orch if k == "orchestrator" else orch.agents[k]; c1, c2 = st.columns([1, 6])
            c1.image(img(f), width=56)
            with c2:
                st.markdown(f"**{ag.name}** <span class='pill'>{ag.status}</span>", unsafe_allow_html=True); st.caption(f"{ag.role}  ·  {ag.contract}")
                st.caption(f"runs: {ag.runs}  ·  last duration: {'%.4fs' % ag.duration if ag.duration else '—'}" + (f"  ·  error: {ag.error}" if ag.error else ""))
        st.subheader("Proof Graph")
        if not orch.graph.nodes: st.info("No recorded events yet. Run any task.")
        else:
            st.graphviz_chart(orch.graph.to_dot(), use_container_width=True)
            for n in reversed(orch.graph.nodes[-60:]):
                with st.expander(f"{n['id']} · {n['kind']} · {n['label']} [{n['status']}]"): st.json(n["data"])
            st.download_button("Export Proof Graph JSON", json.dumps(orch.graph.export(), default=str).encode(), "proof_graph.json")
    elif page == "Task History":
        st.header("Task History")
        if ss.tasks: st.dataframe(pd.DataFrame(ss.tasks), use_container_width=True); st.download_button("History CSV", A.csv_bytes(pd.DataFrame(ss.tasks)), "history.csv")
        else: st.info("No tasks yet in this session. History is not persisted.")
    else:
        st.header("Settings & AI Connection"); rt = router()
        st.write("API key:", "configured" if rt.key else "**missing** — add GEMINI_API_KEY to Streamlit Secrets"); st.write("Models (fallback order):", rt.models)
        if st.button("Test AI Connection", type="primary"):
            try: res = rt.test_connection(); st.success(f"Connected via {res['model']}")
            except R.AIError as e: st.error(str(e))
        st.write("Status:", rt.status); st.caption("Free-tier availability and quotas depend on Google's current policy and your account.")

with side:
    st.subheader("AI Chat"); rt = router(); st.caption(f"Key: {'configured' if rt.key else 'missing'} · Active model: {rt.status['active_model'] or 'none yet'}" + (f" · last error: {rt.status['last_error']}" if rt.status["last_error"] else ""))
    box = st.container(height=420)
    for m in ss.chat[-30:]: box.chat_message(m["role"]).markdown(m["content"])
    if not ss.chat:
        for s in ("Explain my last calculation", "What do my ML metrics mean?"):
            if st.button(s, key="sg" + s): ss.pending = s
    prompt = st.chat_input("Ask NOVEXA") or ss.pop("pending", None)
    if prompt:
        ctx = []
        if ss.trace: ctx.append(f"Last calculator trace: {ss.trace[-6:]}")
        if ss.get("ml"): ctx.append("Last ML result: " + json.dumps({k: ss.ml[k] for k in ("task", "model_name", "model", "baseline", "warnings")}, default=str))
        if ss.get("doc_name"): ctx.append(f"Active document: {ss.doc_name}")
        hist = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in ss.chat[-6:])
        ss.chat.append({"role": "user", "content": prompt})
        out = ask("Context (real results only; never invent how they were obtained):\n" + "\n".join(ctx) + "\n\nConversation:\n" + hist + "\n\nUser: " + prompt)
        ss.chat.append({"role": "assistant", "content": out or "⚠️ Request failed — see the error above, then retry."}); st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("New chat", use_container_width=True): ss.chat = []; st.rerun()
    if c2.button("Retry last", use_container_width=True) and ss.chat and ss.chat[-1]["role"] == "assistant":
        ss.pending = ss.chat[-2]["content"]; ss.chat = ss.chat[:-2]; st.rerun()
