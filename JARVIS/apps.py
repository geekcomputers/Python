import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from .config import APP_INDEX_FILE
from .memory import load_json_list
from .safety import BLOCKED_APPS, DANGEROUS_WORDS
from .text_utils import normalize_text


def safe_search_dirs():
    dirs = [
        Path(os.environ.get("APPDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
        Path(os.environ.get("PROGRAMDATA", "")) / r"Microsoft\Windows\Start Menu\Programs",
        Path(os.environ.get("USERPROFILE", "")) / "Desktop",
        Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "Desktop",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    ]
    return [path for path in dirs if path and path.exists()]


def app_name_from_path(path):
    return re.sub(r"\s+", " ", path.stem.replace(".lnk", "")).strip()


def is_safe_app_path(path):
    suffix = path.suffix.lower()
    if suffix not in {".lnk", ".exe"}:
        return False
    name = normalize_text(app_name_from_path(path))
    parts = {normalize_text(part) for part in path.parts}
    return name not in BLOCKED_APPS and not (parts & BLOCKED_APPS)


def build_application_index():
    apps = {}
    for root in safe_search_dirs():
        try:
            candidates = list(root.rglob("*.lnk"))
            if root.name.lower() in {"program files", "program files (x86)", "programs"}:
                candidates.extend(root.glob("*/*.exe"))
                candidates.extend(root.glob("*.exe"))
        except OSError:
            continue
        for path in candidates:
            if not is_safe_app_path(path):
                continue
            name = app_name_from_path(path)
            key = normalize_text(name)
            apps.setdefault(key, {"name": name, "path": str(path)})
    sorted_apps = sorted(apps.values(), key=lambda item: item["name"].lower())
    APP_INDEX_FILE.write_text(json.dumps(sorted_apps, ensure_ascii=False, indent=2), encoding="utf-8")
    return sorted_apps


def load_application_index(refresh=False):
    if refresh or not APP_INDEX_FILE.exists():
        return build_application_index()
    data = load_json_list(APP_INDEX_FILE)
    return data or build_application_index()


def find_application(name):
    wanted = normalize_text(name)
    if not wanted:
        return None
    if wanted in BLOCKED_APPS or any(word in wanted.split() for word in DANGEROUS_WORDS):
        return None
    normalized_apps = [(app, normalize_text(app.get("name", ""))) for app in load_application_index()]
    for app, app_name in normalized_apps:
        if wanted == app_name:
            return app
    for app, app_name in normalized_apps:
        if wanted in app_name.split():
            return app
    best = None
    best_score = 0
    for app, app_name in normalized_apps:
        if not app_name:
            continue
        if len(wanted) >= 5 and (wanted in app_name or app_name in wanted):
            return app
        score = SequenceMatcher(None, wanted, app_name).ratio()
        if score > best_score:
            best = app
            best_score = score
    return best if best_score >= 0.72 else None


def search_apps(query, limit=8):
    wanted = normalize_text(query)
    scored = []
    for app in load_application_index():
        app_name = normalize_text(app.get("name", ""))
        if not wanted or wanted in app_name:
            score = 1.0
        else:
            score = SequenceMatcher(None, wanted, app_name).ratio()
        if score >= 0.45:
            scored.append((score, app["name"]))
    return [name for _, name in sorted(scored, reverse=True)[:limit]]
