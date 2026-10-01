"""Per-user settings for the desktop app: API keys and the data folder.

Keys are stored in a JSON file in the user's application-data folder
(Windows: %APPDATA%\\ScriptStudio, macOS: ~/Library/Application Support/ScriptStudio,
Linux: ~/.local/share/ScriptStudio) and loaded into the environment at start-up.
"""
from __future__ import annotations

import json
import os
import sys

APP_NAME = 'ScriptStudio'
KEYS = ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY')
PREFS = ('SCRIPT_MODEL', 'TRANSCRIBE_BACKEND', 'WHISPER_MODEL')


def data_dir() -> str:
    if os.getenv('SCRIPT_STUDIO_HOME'):
        d = os.getenv('SCRIPT_STUDIO_HOME')
    elif sys.platform == 'win32':
        d = os.path.join(os.getenv('APPDATA') or os.path.expanduser('~'), APP_NAME)
    elif sys.platform == 'darwin':
        d = os.path.expanduser(f'~/Library/Application Support/{APP_NAME}')
    else:
        d = os.path.join(os.getenv('XDG_DATA_HOME') or os.path.expanduser('~/.local/share'), APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def is_frozen() -> bool:
    return bool(getattr(sys, 'frozen', False))


def _path() -> str:
    return os.path.join(data_dir(), 'settings.json')


def load() -> dict:
    try:
        with open(_path(), encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def apply() -> None:
    """Copy saved settings into the environment (environment variables already set win)."""
    for k, v in load().items():
        if k in KEYS + PREFS and v and not os.getenv(k):
            os.environ[k] = str(v)


def save(updates: dict) -> None:
    cur = load()
    for k, v in updates.items():
        if k not in KEYS + PREFS:
            continue
        v = (v or '').strip()
        if v:
            cur[k] = v
            os.environ[k] = v
        else:
            cur.pop(k, None)
            os.environ.pop(k, None)
    tmp = _path() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cur, f, indent=1)
    os.replace(tmp, _path())
    try:
        os.chmod(_path(), 0o600)
    except OSError:
        pass


def status() -> dict:
    """What the settings screen may show: whether each key is set (never the key itself)."""
    out = {k: bool(os.getenv(k)) for k in KEYS}
    for k in KEYS:
        v = os.getenv(k) or ''
        out[k + '_hint'] = ('…' + v[-4:]) if len(v) > 8 else ''
    for k in PREFS:
        out[k] = os.getenv(k, '')
    return out
