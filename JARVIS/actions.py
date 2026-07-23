import importlib
import os
import webbrowser
from urllib.parse import quote_plus, urlparse

from .ai import ask_model, classify_action
from .apps import find_application
from .config import CMD_OPEN_PHRASES, KNOWN_SITES
from .memory import remember_note
from .safety import BLOCKED_APPS, is_dangerous_request
from .text_utils import normalize_text

try:
    win32con = importlib.import_module("win32con")
    win32gui = importlib.import_module("win32gui")
except ImportError:
    win32con = None
    win32gui = None


def open_application(name):
    app = find_application(name)
    if not app:
        return f"I could not find a safe installed app named {name}."
    path = app["path"]
    os.startfile(path)
    return f"Opening {app['name']}."


def close_visible_window(name):
    if win32gui is None or win32con is None:
        return "Close is not available because pywin32 is missing."
    wanted = normalize_text(name)
    if not wanted or wanted in BLOCKED_APPS:
        return "That close request is blocked for safety."
    matches = []

    def callback(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if wanted in normalize_text(title):
            matches.append((hwnd, title))

    win32gui.EnumWindows(callback, None)
    if not matches:
        return f"I could not find an open window matching {name}."
    hwnd, title = matches[0]
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    return f"Closing {title or name}."


def is_safe_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def open_web_target(target):
    cleaned = normalize_text(target)
    url = KNOWN_SITES.get(cleaned)
    if not url and "." in cleaned:
        url = target if target.startswith(("http://", "https://")) else f"https://{target}"
    if not url or not is_safe_url(url):
        return "I can only open safe web addresses."
    webbrowser.open(url)
    return f"Opening {url}."


def run_action(action):
    if not action:
        return ""
    action = action.strip()
    lowered = action.lower()
    if lowered == "open_cmd":
        os.startfile("cmd.exe")
        return "Opening Command Prompt."
    if lowered == "blocked" or is_dangerous_request(action):
        return "I cannot do that for safety."
    if lowered.startswith("open_web:"):
        return open_web_target(action.split(":", 1)[1].strip())
    if lowered.startswith("search_google:"):
        query = action.split(":", 1)[1].strip().strip("<> ")
        if not query or is_dangerous_request(query):
            return "I cannot search that for safety."
        webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
        return f"Searching Google for {query}."
    if lowered.startswith("open_app:"):
        return open_application(action.split(":", 1)[1].strip())
    if lowered.startswith("close_app:"):
        return close_visible_window(action.split(":", 1)[1].strip())
    if lowered == "chat":
        return ""
    return ""


def rule_based_action(text):
    cleaned = normalize_text(text)
    if not cleaned:
        return ""
    if cleaned in CMD_OPEN_PHRASES:
        return "open_cmd"
    if is_dangerous_request(cleaned):
        return "blocked"

    search_prefixes = ["search for ", "google search ", "look up ", "find ", "ara ", "google da ara "]
    for prefix in search_prefixes:
        if cleaned.startswith(prefix):
            query = cleaned.removeprefix(prefix).strip()
            return f"search_google:{query}" if query else ""
    if cleaned.endswith(" ara"):
        query = cleaned[: -len(" ara")].strip()
        return f"search_google:{query}" if query else ""

    close_prefixes = ["close ", "kapat ", "close the ", "can you close "]
    for prefix in close_prefixes:
        if cleaned.startswith(prefix):
            app = cleaned.removeprefix(prefix).strip()
            return f"close_app:{app}" if app else ""
    if cleaned.endswith(" kapat"):
        app = cleaned[: -len(" kapat")].strip()
        return f"close_app:{app}" if app else ""

    open_prefixes = ["open ", "launch ", "start ", "can you open ", "please open ", "ac ", "aç "]
    suffix_open_words = [" ac", " aç", " i ac", " i aç", " u ac", " u aç"]
    for site, url in KNOWN_SITES.items():
        if cleaned in {site, f"open {site}", f"{site} ac", f"{site} aç"}:
            return f"open_web:{url}"
    for prefix in open_prefixes:
        if cleaned.startswith(prefix):
            target = cleaned.removeprefix(prefix).strip()
            if target in KNOWN_SITES:
                return f"open_web:{target}"
            return f"open_app:{target}" if target else ""
    for suffix in suffix_open_words:
        if cleaned.endswith(suffix):
            target = cleaned[: -len(suffix)].strip()
            if target in KNOWN_SITES:
                return f"open_web:{target}"
            return f"open_app:{target}" if target else ""
    return ""


def handle_user_text(text):
    cleaned = normalize_text(text)
    if cleaned.startswith("remember that "):
        return remember_note(text.split("remember that", 1)[1].strip())
    action = rule_based_action(text) or classify_action(text)
    if action.strip().lower() == "blocked" and not is_dangerous_request(text):
        action = "chat"
    answer = run_action(action)
    if answer:
        return answer
    return ask_model(text)
