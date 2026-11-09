import hashlib
import json
import re
import time

import httpx

from .models import ROOT, Model

CACHE = ROOT / "cache"
THINK = re.compile(r"<(think|thought)>.*?</\1>", re.S)
RETRIES = 40
RETRY_AFTER = re.compile(r"try again in ([\d.]+)s")


def _cache_path(key_material):
    digest = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
    return CACHE / digest[:2] / f"{digest}.json"


def _post(model, path, body):
    delay = 2.0
    for attempt in range(RETRIES):
        try:
            r = httpx.post(
                model.base_url + path,
                json=body,
                headers={"Authorization": f"Bearer {model.key}"},
                timeout=300,
            )
        except httpx.TransportError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if r.status_code == 200:
            return r.json()
        retryable = r.status_code in (408, 429) or r.status_code >= 500
        if not retryable or attempt == RETRIES - 1:
            raise RuntimeError(f"{model.name} HTTP {r.status_code}: {r.text[:300]}")
        hint = RETRY_AFTER.search(r.text)
        time.sleep(float(hint.group(1)) + 1 if hint else delay)
        delay = min(delay * 2, 60)


def _request(model: Model, messages, max_tokens):
    sampling = {} if model.pinned else {"temperature": 0, "max_tokens": max(max_tokens, model.min_tokens)}
    return {"model": model.id, "messages": messages, **sampling, **model.extra}


def cached(model: Model, messages, max_tokens=1024):
    """The stored reply for this request, or None if it was never completed."""
    cp = _cache_path([model.base_url, "/chat/completions", _request(model, messages, max_tokens)])
    return json.loads(cp.read_text())["text"] if cp.exists() else None


def complete(model: Model, messages, max_tokens=1024):
    """Greedy completion, cached on disk by request so reruns cost nothing."""
    req = _request(model, messages, max_tokens)
    path = "/chat/completions"
    cp = _cache_path([model.base_url, path, req])
    if cp.exists():
        return json.loads(cp.read_text())["text"]
    data = _post(model, path, req)
    out = THINK.sub("", data["choices"][0]["message"].get("content") or "").strip()
    if not out:
        return out
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"text": out}, ensure_ascii=False))
    return out
