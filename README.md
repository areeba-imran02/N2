# NOVEXA AI — Beyond Answers. Into Action.
Streamlit app: safe scientific calculator, scikit-learn ML Studio, statistics lab, PDF retrieval with page references, Gemini chat with model fallback.

## Run locally
    pip install -r requirements.txt
    mkdir -p .streamlit && echo 'GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"' > .streamlit/secrets.toml   # never commit
    streamlit run app.py
    python run_tests.py    # or: pytest -q

## Deploy (Streamlit Community Cloud)
Push to GitHub (no secrets) → deploy → App settings → Secrets → add `GEMINI_API_KEY = "..."` → rerun → Settings page → Test AI Connection.
Optional: `GEMINI_MODELS = '["gemini-2.5-flash","gemini-2.5-flash-lite"]'`. Verify IDs and free-tier status in Google's docs; quotas vary by account and can change.

## Data handling
Datasets are processed locally. Gemini only receives the prompt, retrieved PDF excerpts, and the last calculator trace. Storage on Community Cloud is ephemeral.

## Features
Agent Control Center (7 agents + Orchestrator, real status/timings), Proof Graph (Graphviz + event inspector, JSON export), calculator (keypad, DEG/RAD, symbolic, matrix, trace), PDF Lab (page-aware QA, summary, quiz, similarity), ML Studio (classification, regression, K-Means+PCA, Isolation Forest, downloads), Statistics Lab, Coding Lab (static analysis only), right-side Gemini chat, TXT/DOCX/PDF reports.

## Testing
`pip install -r requirements.txt && pytest -q` (or `python run_tests.py` without pytest). Gemini is mocked in tests; no key needed.
See DEPLOYMENT.md for deployment.
