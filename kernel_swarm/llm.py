"""Tiny provider-agnostic LLM client. Standard library only, no SDK installs.

Configure in a `.env` file at the repo root (gitignored, never commit it):

    # pick ONE provider
    ANTHROPIC_API_KEY=...            # Claude
    OPENAI_API_KEY=...               # OpenAI
    GEMINI_API_KEY=...               # Google Gemini (OpenAI-compatible endpoint)
    LLM_BASE_URL=http://localhost:8000/v1   # local vLLM server (OpenAI-compatible)

    LLM_MODEL=...                    # optional for Claude, required for the others
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


def load_env(path=REPO_ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _post(url, headers, body, timeout=600, retries=4):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                wait = 2 ** attempt * 5
                print(f"   LLM API {e.code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM API error {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            raise RuntimeError(f"Could not reach LLM API: {e.reason}") from None


class LLM:
    def __init__(self):
        load_env()
        self.model = os.environ.get("LLM_MODEL")
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.provider = "anthropic"
            self.model = self.model or DEFAULT_CLAUDE_MODEL
        elif os.environ.get("OPENAI_API_KEY"):
            self.provider, self.base, self.key = "openai", "https://api.openai.com/v1", os.environ["OPENAI_API_KEY"]
        elif os.environ.get("GEMINI_API_KEY"):
            self.provider, self.base, self.key = "gemini", GEMINI_OPENAI_URL, os.environ["GEMINI_API_KEY"]
        elif os.environ.get("LLM_BASE_URL"):
            self.provider, self.base, self.key = "local", os.environ["LLM_BASE_URL"].rstrip("/"), os.environ.get("LLM_API_KEY", "none")
        else:
            raise SystemExit(
                "No LLM configured. Create a .env file in the repo root with one of:\n"
                "  ANTHROPIC_API_KEY=...   OPENAI_API_KEY=...   GEMINI_API_KEY=...   LLM_BASE_URL=...\n"
                "(see .env.example)")
        if not self.model:
            raise SystemExit(f"Set LLM_MODEL in .env for provider '{self.provider}'.")

    def __repr__(self):
        return f"LLM({self.provider}:{self.model})"

    def chat(self, system, user, max_tokens=8000):
        if self.provider == "anthropic":
            out = _post(
                "https://api.anthropic.com/v1/messages",
                {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
                {"model": self.model, "max_tokens": max_tokens, "system": system,
                 "messages": [{"role": "user", "content": user}]},
            )
            return "".join(b.get("text", "") for b in out.get("content", []))
        out = _post(
            f"{self.base}/chat/completions",
            {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            {"model": self.model,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
        return out["choices"][0]["message"]["content"] or ""
