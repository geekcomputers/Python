import json
from urllib.request import Request, urlopen

import speech_recognition as sr

from . import state
from .actions import handle_user_text, rule_based_action
from .apps import build_application_index, search_apps
from .config import OPENAI_BASE_URL, OPENAI_MODEL
from .memory import clear_memory
from .speech import init_tts_engine, listen_once, say, set_tts_engine
from .text_utils import normalize_text


def should_exit(text):
    return normalize_text(text) in {"exit", "quit", "bye", "cik", "çık", "stop jarvis", "kapat jarvis"}


def wants_type_mode(text):
    return normalize_text(text) in {
        "type",
        "type mode",
        "keyboard",
        "keyboard mode",
        "yaz",
        "yazi",
        "yazı",
        "yazi modu",
        "yazı modu",
    }


def wants_developer_mode(text):
    return normalize_text(text) in {"developer mode", "development mode", "dev mode", "change mode"}


def wants_normal_mode(text):
    return normalize_text(text) in {"normal mode", "user mode", "exit developer mode"}


def choose_best_candidate(candidates):
    if not candidates:
        return ""
    for candidate in candidates:
        if rule_based_action(candidate):
            return candidate
    return candidates[0]


def help_text():
    return (
        "Commands: say an app/site to open it, say 'python ara' or 'search for python', "
        "'chrome kapat' to close a visible window, 'remember that ...' to save a note, "
        "'developer mode' to show prompts/raw AI/token usage, '/apps chrome' in type mode to list apps."
    )


def check_lm_studio():
    try:
        request = Request(f"{OPENAI_BASE_URL}/models")
        with urlopen(request, timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = [item.get("id", "") for item in data.get("data", [])]
        if OPENAI_MODEL in models:
            say(f"LM Studio connected: {OPENAI_MODEL}")
        else:
            say("LM Studio is running, but the selected model was not listed.")
    except Exception as exc:
        say(f"LM Studio not ready: {exc}")


def handle_command(user_input):
    cleaned = normalize_text(user_input)
    if wants_developer_mode(user_input):
        state.set_developer_mode(True)
        return "Developer mode enabled. I will print prompts, raw AI outputs, and token usage when available."
    if wants_normal_mode(user_input):
        state.set_developer_mode(False)
        return "Developer mode disabled."
    if cleaned in {"help", "/help", "yardim", "yardım"}:
        return help_text()
    if cleaned in {"clear memory", "memory clear", "hafizayi temizle", "hafızayı temizle"}:
        clear_memory()
        return "Memory cleared."
    if cleaned.startswith("/apps"):
        query = user_input[5:].strip()
        matches = search_apps(query)
        return "Matching apps: " + (", ".join(matches) if matches else "none found")
    return handle_user_text(user_input)


def handle_and_say(user_input):
    try:
        say(handle_command(user_input))
    except Exception as exc:
        say(f"I could not complete that: {exc}")


def voice_loop():
    say("Ready.")
    print("Listening stays on. Speak when you want something, or say 'type mode' to write.")
    check_lm_studio()
    apps = build_application_index()
    say(f"Safe app index ready: {len(apps)} apps found.")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2
    recognizer.non_speaking_duration = 1

    try:
        with sr.Microphone() as source:
            print("Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while True:
                user_input = listen_once(recognizer, source, choose_best_candidate)
                if not user_input:
                    print("No clear speech detected. Continuing to listen...")
                    continue
                if should_exit(user_input):
                    say("Goodbye.")
                    break
                if wants_type_mode(user_input):
                    typed = input("Type to Jarvis: ").strip()
                    if typed:
                        print(f"You typed: {typed}")
                        if should_exit(typed):
                            say("Goodbye.")
                            break
                        handle_and_say(typed)
                    print("Listening again...")
                    continue
                if normalize_text(user_input) in {"refresh apps", "uygulamalari yenile", "uygulamaları yenile"}:
                    apps = build_application_index()
                    say(f"Safe app index refreshed: {len(apps)} apps found.")
                    continue
                handle_and_say(user_input)
    except KeyboardInterrupt:
        print()
        say("Goodbye.")
    except Exception as exc:
        say(f"Microphone error: {exc}")
        print("Falling back to type mode.")
        typed_loop()


def typed_loop():
    say("Type mode is active. Type /voice to return to listening.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            say("Goodbye.")
            break
        if not user_input:
            continue
        if should_exit(user_input):
            say("Goodbye.")
            break
        if normalize_text(user_input) in {"/voice", "voice", "listen", "listening"}:
            voice_loop()
            break
        if normalize_text(user_input) in {"/refresh", "refresh apps"}:
            apps = build_application_index()
            say(f"Safe app index refreshed: {len(apps)} apps found.")
            continue
        handle_and_say(user_input)


def main(argv=None):
    argv = argv or []
    set_tts_engine(init_tts_engine())
    if len(argv) > 1 and argv[1].lower() in {"--type", "--text"}:
        typed_loop()
    else:
        voice_loop()
