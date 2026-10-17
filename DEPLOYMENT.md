# Deploying NOVEXA AI on Streamlit Community Cloud
1. Push this folder to a GitHub repo. `.gitignore` already excludes `.streamlit/secrets.toml`; never commit a real key.
2. share.streamlit.io → New app → pick repo, branch, main file `app.py`.
3. App → Settings → Secrets → paste:
       GEMINI_API_KEY = "your key from Google AI Studio"
       GEMINI_MODELS = '["gemini-2.5-flash", "gemini-2.5-flash-lite"]'
4. Save; the app restarts. Open **Settings & AI Connection → Test AI Connection**.
5. Local: `cp .streamlit/secrets.toml.example .streamlit/secrets.toml`, edit, `pip install -r requirements.txt`, `streamlit run app.py`.
Notes: filesystem and session state are ephemeral; nothing is persisted. Free-tier models/quotas are set by Google and can change; exhausted quota shows an error, never a fabricated answer. Code execution is disabled by design. Scanned PDFs are rejected (no OCR).
