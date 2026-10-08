"""Provider wrapper. API keys are read from environment variables ONLY and never logged."""
import os, re, json, time

class LLMError(Exception):
    pass

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o", "mock": "mock-rule-based"}

class LLMClient:
    def __init__(self, provider=None, model=None, temperature=None, max_tokens=800, retries=3):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        if self.provider not in DEFAULT_MODELS:
            raise LLMError(f"Unsupported provider '{self.provider}'.")
        self.model = model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0")) if temperature is None else float(temperature)
        self.temperature_applied = True
        self.max_tokens = max_tokens
        self.retries = retries
        self.model_returned = None
        self._client = None

    @property
    def is_mock(self):
        return self.provider == "mock"

    def key_present(self):
        env = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "mock": None}[self.provider]
        return True if env is None else bool(os.getenv(env))

    def _get(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI()
        return self._client

    def complete(self, system: str, user: str) -> str:
        if self.is_mock:
            return _mock(system, user)
        if not self.key_present():
            raise LLMError("API key not found in environment variables.")
        last = None
        for attempt in range(self.retries):
            try:
                return self._call(system, user)
            except Exception as e:  # noqa
                msg = str(e)
                if "temperature" in msg.lower() and self.temperature_applied:
                    self.temperature_applied = False  # model rejects temperature; retry without it and record this
                    continue
                last = f"{type(e).__name__}: {msg[:300]}"
                time.sleep(2 ** attempt)
        raise LLMError(last or "Unknown API failure")

    def _call(self, system, user):
        c = self._get()
        if self.provider == "anthropic":
            kw = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
            if self.temperature_applied:
                kw["temperature"] = self.temperature
            r = c.messages.create(**kw)
            self.model_returned = getattr(r, "model", None)
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self.temperature_applied:
            kw["temperature"] = self.temperature
        r = c.chat.completions.create(**kw)
        self.model_returned = getattr(r, "model", None)
        return r.choices[0].message.content or ""

# ---- Mock provider: deterministic, rule-based. FOR SOFTWARE TESTING ONLY, never for reported results. ----
_MOCK_TERMS = {"chairman": "chairperson", "policeman": "police officer", "mankind": "humankind",
               "manpower": "workforce", "fireman": "firefighter", "stewardess": "flight attendant",
               "salesman": "salesperson", "businessman": "businessperson"}

def _mock(system, user):
    m = re.search(r"<<<\n(.*?)\n>>>", user, re.S)
    text = m.group(1) if m else user
    hits = [t for t in _MOCK_TERMS if re.search(rf"\b{t}\b", text, re.I)]
    if "[TASK:DETECT]" in system:
        if "FORCE_API_ERROR" in text:
            raise LLMError("mock api failure")
        if "FORCE_BAD_JSON" in text:
            return "not json"
        return json.dumps({"is_biased": bool(hits), "category": "Gendered occupational language" if hits else "None",
                           "explanation": f"Uses gendered term '{hits[0]}'." if hits else "No gendered wording found.",
                           "flagged_phrase": hits[0] if hits else "", "confidence": 0.9})
    out = text
    for t in hits:
        out = re.sub(rf"\b{t}\b", _MOCK_TERMS[t], out, flags=re.I)
    return json.dumps({"alternative": out, "changes": ", ".join(f"{t}->{_MOCK_TERMS[t]}" for t in hits)})
