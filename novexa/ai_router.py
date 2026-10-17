"""Gemini router: error classification, bounded retries, fallback, cooldowns. SDK imported lazily."""
import random, time
DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]  # editable; set GEMINI_MODELS in secrets. Verify against Google docs.

class AIError(Exception): pass
class NotConfigured(AIError): pass
class AuthError(AIError): pass
class AllModelsUnavailable(AIError): pass

def classify(exc):
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if code in (401, 403) or "api key" in msg or "permission" in msg: return "auth"
    if code == 429 or "quota" in msg or "resource_exhausted" in msg: return "rate_limit"
    if code in (500, 502, 503, 504) or "unavailable" in msg or "overloaded" in msg: return "retryable"
    if code == 404: return "model_missing"
    return "invalid"

class Router:
    def __init__(self, api_key, models=None, call=None, sleep=time.sleep, clock=time.monotonic, retries=2, budget_s=45, cooldown_s=60):
        self.key, self.models = api_key, list(models or DEFAULT_MODELS)
        self._call, self.sleep, self.clock = call, sleep, clock
        self.retries, self.budget, self.cooldown = retries, budget_s, cooldown_s
        self.cool = {}; self.status = {"active_model": None, "attempts": [], "last_error": None}

    def _default_call(self, model, prompt):
        from google import genai
        return genai.Client(api_key=self.key).models.generate_content(model=model, contents=prompt).text

    def generate(self, prompt):
        if not self.key: raise NotConfigured("GEMINI_API_KEY is missing. Add it in Streamlit Secrets.")
        call = self._call or self._default_call
        start = self.clock(); self.status["attempts"] = []
        for m in self.models:
            if self.cool.get(m, 0) > self.clock(): self.status["attempts"].append((m, "cooling_down")); continue
            for i in range(self.retries + 1):
                if self.clock() - start > self.budget: raise AllModelsUnavailable("time budget exhausted")
                try:
                    out = call(m, prompt)
                    self.status.update(active_model=m, last_error=None); self.status["attempts"].append((m, "ok"))
                    return out, m
                except Exception as e:
                    kind = classify(e); self.status["attempts"].append((m, kind)); self.status["last_error"] = f"{kind}: {type(e).__name__}"
                    if kind == "auth": raise AuthError("Gemini rejected the API key. Check the secret.") from e
                    if kind == "invalid": raise AIError(f"Invalid request: {type(e).__name__}") from e
                    if kind == "rate_limit": self.cool[m] = self.clock() + self.cooldown; break
                    if kind == "model_missing": break
                    if i < self.retries: self.sleep(min(8, 2 ** i) + random.random())
        raise AllModelsUnavailable("All configured models are rate-limited or unavailable. Free quotas may be exhausted; retry later.")

    def test_connection(self):
        text, model = self.generate("Reply with the single word: ok")
        return {"ok": True, "model": model, "reply": text.strip()[:40]}
