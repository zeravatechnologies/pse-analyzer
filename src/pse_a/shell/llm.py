"""Language models for the PSE-L conversation -- many providers, one interface.

Three kinds of access:
  anthropic      -- Claude, through the OFFICIAL SDK (pip anthropic)
  openai_compat  -- everything that speaks the OpenAI /chat/completions
                    protocol: OpenAI, Gemini (Google's compatible endpoint),
                    Groq, Mistral, DeepSeek, xAI, OpenRouter (hundreds of models
                    behind one key), Together, Fireworks, Cerebras, DeepInfra,
                    NVIDIA, Hugging Face, Cohere, SambaNova, Moonshot,
                    Perplexity... plus local servers (Ollama, LM Studio, the
                    llama.cpp server, vLLM) and an arbitrary endpoint
  local_gguf     -- a GGUF directly through our own runtime; works with no
                    internet at all (the minimum)

Model names are NOT hard-coded (apart from the default Claude): once a key has
been entered the model is picked from the list the provider itself returns
(/models), so the list never goes stale.

Keys: ~/.config/pse-a/llm.json, mode 0600. They are never logged and never sent
to anyone other than the provider they belong to. Only SUMMARIES of results
(the stories and a few numbers) ever leave the machine -- never the model, the
weights or the activations.
"""
from __future__ import annotations

import json
import os
import stat
import urllib.error
import urllib.request
from pathlib import Path

CLAUDE_DEFAULT = "claude-opus-5"

# id: (display name, group, kind, base url, environment variable, where to get a key)
# NOTE: the group values ("online", "lokalno", "prilagodjeno") are LOOKUP KEYS --
# web/app.py indexes a dict literally keyed by them. They are never shown to the
# user, and the visible group labels live on the app.py side. Do not translate them.
PROVIDERS: dict[str, tuple] = {
    "anthropic":   ("Claude (Anthropic)", "online", "anthropic", "", "ANTHROPIC_API_KEY",
                    "https://console.anthropic.com/settings/keys"),
    "openai":      ("ChatGPT / OpenAI", "online", "openai_compat", "https://api.openai.com/v1",
                    "OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "gemini":      ("Gemini (Google)", "online", "openai_compat",
                    "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
                    "https://aistudio.google.com/apikey"),
    "groq":        ("Groq", "online", "openai_compat", "https://api.groq.com/openai/v1",
                    "GROQ_API_KEY", "https://console.groq.com/keys"),
    "mistral":     ("Mistral", "online", "openai_compat", "https://api.mistral.ai/v1",
                    "MISTRAL_API_KEY", "https://console.mistral.ai/api-keys"),
    "deepseek":    ("DeepSeek", "online", "openai_compat", "https://api.deepseek.com/v1",
                    "DEEPSEEK_API_KEY", "https://platform.deepseek.com/api_keys"),
    "xai":         ("Grok (xAI)", "online", "openai_compat", "https://api.x.ai/v1",
                    "XAI_API_KEY", "https://console.x.ai"),
    "openrouter":  ("OpenRouter (hundreds of models)", "online", "openai_compat",
                    "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
                    "https://openrouter.ai/keys"),
    "together":    ("Together AI", "online", "openai_compat", "https://api.together.xyz/v1",
                    "TOGETHER_API_KEY", "https://api.together.ai/settings/api-keys"),
    "fireworks":   ("Fireworks AI", "online", "openai_compat",
                    "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
                    "https://fireworks.ai/account/api-keys"),
    "cerebras":    ("Cerebras", "online", "openai_compat", "https://api.cerebras.ai/v1",
                    "CEREBRAS_API_KEY", "https://cloud.cerebras.ai"),
    "deepinfra":   ("DeepInfra", "online", "openai_compat", "https://api.deepinfra.com/v1/openai",
                    "DEEPINFRA_API_KEY", "https://deepinfra.com/dash/api_keys"),
    "nvidia":      ("NVIDIA NIM", "online", "openai_compat", "https://integrate.api.nvidia.com/v1",
                    "NVIDIA_API_KEY", "https://build.nvidia.com"),
    "huggingface": ("Hugging Face", "online", "openai_compat", "https://router.huggingface.co/v1",
                    "HF_TOKEN", "https://huggingface.co/settings/tokens"),
    "cohere":      ("Cohere", "online", "openai_compat", "https://api.cohere.ai/compatibility/v1",
                    "COHERE_API_KEY", "https://dashboard.cohere.com/api-keys"),
    "sambanova":   ("SambaNova", "online", "openai_compat", "https://api.sambanova.ai/v1",
                    "SAMBANOVA_API_KEY", "https://cloud.sambanova.ai"),
    "moonshot":    ("Kimi (Moonshot)", "online", "openai_compat", "https://api.moonshot.ai/v1",
                    "MOONSHOT_API_KEY", "https://platform.moonshot.ai"),
    "perplexity":  ("Perplexity", "online", "openai_compat", "https://api.perplexity.ai",
                    "PERPLEXITY_API_KEY", "https://www.perplexity.ai/settings/api"),
    "ollama":      ("Ollama (local)", "lokalno", "openai_compat", "http://localhost:11434/v1",
                    "", "https://ollama.com"),
    "lmstudio":    ("LM Studio (local)", "lokalno", "openai_compat", "http://localhost:1234/v1",
                    "", "https://lmstudio.ai"),
    "llamacpp":    ("llama.cpp server (local)", "lokalno", "openai_compat",
                    "http://localhost:8080/v1", "", "https://github.com/ggml-org/llama.cpp"),
    "vllm":        ("vLLM (local)", "lokalno", "openai_compat", "http://localhost:8000/v1",
                    "", "https://docs.vllm.ai"),
    "local_gguf":  ("GGUF file, no server (built in)", "lokalno", "local_gguf", "", "", ""),
    "custom":      ("Another endpoint (OpenAI-compatible)", "prilagođeno", "openai_compat", "",
                    "", ""),
}


class LLMError(Exception):
    pass


# ------------------------------------------------------------------ settings
def config_path() -> Path:
    from .osenv import config_dir
    return config_dir() / "llm.json"


def load_config() -> dict:
    p = config_path()
    if not p.exists():
        return {"active": None, "providers": {}, "dismissed": False}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {"active": None, "providers": {}, "dismissed": False}


def save_config(cfg: dict) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(tmp, p)
    from .osenv import harden_file
    harden_file(p)


def entry(cfg: dict, pid: str) -> dict:
    e = dict(cfg.get("providers", {}).get(pid, {}))
    _, _, kind, base, env, _ = PROVIDERS[pid]
    e.setdefault("base_url", base)
    if not e.get("key") and env:
        e["key"] = os.environ.get(env, "")
    return e


def masked(key: str) -> str:
    return "" if not key else (key[:4] + "…" + key[-4:] if len(key) > 12 else "••••")


# ------------------------------------------------------------------ OpenAI-compatible
def _http(url: str, key: str, body: dict | None = None, timeout: float = 60.0):
    headers = {"Content-Type": "application/json", "User-Agent": "pse-a"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        raise LLMError(f"HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"no connection to {url.split('/')[2]}: {e.reason}") from None


def _oa_models(base: str, key: str) -> list[str]:
    with _http(base.rstrip("/") + "/models", key, timeout=20) as r:
        d = json.loads(r.read())
    items = d.get("data", d.get("models", d if isinstance(d, list) else []))
    out = []
    for m in items:
        mid = m.get("id") or m.get("name") if isinstance(m, dict) else m
        if mid:
            out.append(str(mid).removeprefix("models/"))
    return sorted(set(out))


def _oa_chat(base: str, key: str, model: str, system: str, messages: list[dict],
             max_tokens: int = 2048):
    local = any(h in base for h in ("localhost", "127.0.0.1"))
    if local:
        # a local model on the CPU: 2048 tokens is half an hour, and the server
        # (Ollama) handles one request at a time -- a long answer blocks the next
        max_tokens = min(max_tokens, 600)
    body = {"model": model, "stream": True, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}] + messages}
    # a local model on a weak processor can stay silent for minutes before the first token
    tmo = 1200 if local else 300
    try:
        yield from _oa_stream(base, key, body, tmo)
    except (TimeoutError, OSError) as e:
        raise LLMError(f"the model did not answer in time ({type(e).__name__})") from None


def _oa_stream(base, key, body, tmo):
    with _http(base.rstrip("/") + "/chat/completions", key, body, timeout=tmo) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                return
            try:
                d = json.loads(chunk)
            except ValueError:
                continue
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    yield piece


# ------------------------------------------------------------------ Claude (the official SDK)
def _claude_client(key: str):
    try:
        import anthropic
    except ImportError:
        raise LLMError("the official Claude SDK is not installed "
                       "(pip install anthropic in the pse-a venv)") from None
    return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()


def _claude_models(key: str) -> list[str]:
    c = _claude_client(key)
    try:
        return [m.id for m in c.models.list()]
    except Exception as e:                       # noqa: BLE001 -- translated for the human
        raise LLMError(_claude_err(e)) from None


def _claude_err(e) -> str:
    import anthropic
    if isinstance(e, anthropic.AuthenticationError):
        return "the key was not accepted"
    if isinstance(e, anthropic.PermissionDeniedError):
        return "the key has no permission for this model"
    if isinstance(e, anthropic.NotFoundError):
        return "the model does not exist, or is not available to this key"
    if isinstance(e, anthropic.RateLimitError):
        return "too many requests -- try again in a minute"
    if isinstance(e, anthropic.APIConnectionError):
        return "no connection to the Anthropic API"
    if isinstance(e, anthropic.APIStatusError):
        return f"server error ({e.status_code})"
    return str(e)


def _claude_chat(key: str, model: str, system: str, messages: list[dict],
                 max_tokens: int = 16000):
    """Streaming (so longer answers do not hit the timeout). On claude-opus-5 and
    Fable 5.1 the server-side fallbacks are enabled: if a safety classifier
    refuses the request, the API re-runs it on the recommended model itself
    instead of returning a refusal."""
    import anthropic
    c = _claude_client(key)
    kw = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
    if model in ("claude-opus-5", "claude-fable-5-1"):
        stream_ctx = c.beta.messages.stream(betas=["server-side-fallback-2026-07-01"],
                                            extra_body={"fallbacks": "default"}, **kw)
    else:
        stream_ctx = c.messages.stream(**kw)
    try:
        with stream_ctx as stream:
            for text in stream.text_stream:
                yield text
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n(The model refused to answer this question.)"
        elif final.stop_reason == "max_tokens":
            yield "\n\n(The answer was cut off at the length limit.)"
    except anthropic.APIError as e:
        raise LLMError(_claude_err(e)) from None


# ------------------------------------------------------------------ local GGUF
_LOCAL: dict = {}


def _local_chat(path: str, system: str, messages: list[dict], max_new: int = 400):
    from ..core.runtime.qwen2 import Qwen2Runtime, generate, chat_prompt
    if not path or not Path(path).exists():
        raise LLMError("choose an existing GGUF file (qwen2 instruct)")
    rt = _LOCAL.get(path)
    if rt is None:
        _LOCAL.clear()
        rt = _LOCAL[path] = Qwen2Runtime(path)
    turns = [(m["role"], m["content"]) for m in messages]
    ids = chat_prompt(rt, system, turns)
    stop = tuple(i for i in (rt.tok.token_to_id("<|im_end|>"),
                             rt.tok.token_to_id("<|endoftext|>")) if i is not None)
    buf = []
    for tok in generate(rt, ids, max_new=max_new, temperature=0.4, stop_ids=stop):
        buf.append(tok)
        text = rt.decode(buf)
        if not text.endswith("�"):          # do not send half of a UTF-8 character
            yield text
            buf = []


# ------------------------------------------------------------------ the public interface
def list_models(pid: str, e: dict) -> list[str]:
    kind = PROVIDERS[pid][2]
    if kind == "anthropic":
        return _claude_models(e.get("key", ""))
    if kind == "local_gguf":
        return []
    if not e.get("base_url"):
        raise LLMError("enter the address of the server")
    return _oa_models(e["base_url"], e.get("key", ""))


def chat(pid: str, e: dict, system: str, messages: list[dict]):
    kind = PROVIDERS[pid][2]
    if kind == "anthropic":
        yield from _claude_chat(e.get("key", ""), e.get("model") or CLAUDE_DEFAULT,
                                system, messages)
    elif kind == "local_gguf":
        yield from _local_chat(e.get("gguf", ""), system, messages)
    else:
        if not e.get("model"):
            raise LLMError("choose a model")
        yield from _oa_chat(e["base_url"], e.get("key", ""), e["model"], system, messages)
