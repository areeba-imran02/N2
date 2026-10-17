import pytest
from novexa import ai_router
class Response:
    text="ok"
class Models:
    def __init__(self, fail_first=False): self.calls=[]; self.fail_first=fail_first
    def generate_content(self,model,contents):
        self.calls.append(model)
        if self.fail_first and len(self.calls)==1: raise RuntimeError("503 unavailable")
        return Response()
class Client:
    def __init__(self, fail_first=False): self.models=Models(fail_first)
def test_fallback(monkeypatch):
    c=Client(True)
    r=ai_router.generate_text("hi",models=["m1","m2"],max_attempts_per_model=1,client=c,sleeper=lambda _:None)
    assert r["model"]=="m2"
def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    with pytest.raises(ai_router.AIConfigurationError):
        ai_router.generate_text("hi",client=None)
