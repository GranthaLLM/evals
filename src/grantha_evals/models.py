import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Model:
    name: str
    id: str
    base_url: str
    key: str
    extra: dict = field(default_factory=dict)
    pinned: bool = False
    min_tokens: int = 0


def _key(env, file):
    if os.environ.get(env):
        return os.environ[env]
    path = Path(file).expanduser()
    if path.exists():
        return path.read_text().strip()
    raise SystemExit(f"no API key: set {env} or write {path}")


def load(name, config=ROOT / "models.toml"):
    cfg = tomllib.loads(config.read_text())
    for m in cfg["model"]:
        if m["name"] == name:
            merged = {**cfg["defaults"], **m}
            return Model(
                name=name,
                id=merged["id"],
                base_url=merged["base_url"],
                key=_key(merged["key_env"], merged["key_file"]),
                extra=merged.get("extra", {}),
                pinned=merged.get("pinned", False),
                min_tokens=merged.get("min_tokens", 0),
            )
    raise SystemExit(f"unknown model {name!r}, known: {[m['name'] for m in cfg['model']]}")


def names(config=ROOT / "models.toml"):
    return [m["name"] for m in tomllib.loads(config.read_text())["model"]]
