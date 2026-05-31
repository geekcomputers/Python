import importlib
import os
import sys
import tempfile

import speech_recognition as sr

from .config import LISTEN_PHRASE_SECONDS, SPEECH_LANGUAGES, TTS_MODE
from .text_utils import clean_assistant_output, normalize_text

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from gtts import gTTS
    from playsound import playsound
except ImportError:
    gTTS = None
    playsound = None

try:
    win32com_client = importlib.import_module("win32com.client")
except ImportError:
    win32com_client = None

TTS_ENGINE = None
TTS_DISABLED = False


def init_tts_engine():
    if win32com_client is not None:
        try:
            voice = win32com_client.Dispatch("SAPI.SpVoice")
            voices = voice.GetVoices()
            if voices.Count:
                voice.Voice = voices.Item(0)
            return voice
        except Exception as exc:
            print(f"SAPI voice disabled: {exc}")
    if pyttsx3 is None:
        return None
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        engine.setProperty("rate", 165)
        return engine
    except Exception as exc:
        print(f"TTS engine disabled: {exc}")
        return None


def set_tts_engine(engine):
    global TTS_ENGINE
    TTS_ENGINE = engine


def speak(text):
    global TTS_DISABLED
    if TTS_DISABLED:
        return
    if TTS_MODE == "none":
        return
    if TTS_ENGINE is not None:
        try:
            if hasattr(TTS_ENGINE, "Speak"):
                TTS_ENGINE.Speak(str(text))
            else:
                TTS_ENGINE.say(str(text))
                TTS_ENGINE.runAndWait()
            return
        except Exception as exc:
            print(f"TTS engine failed: {exc}")
            TTS_DISABLED = True
            return
    if gTTS is None or playsound is None:
        return
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_path = temp_audio.name
        gTTS(text=str(text), lang="en", slow=False).save(temp_path)
        playsound(temp_path)
    except Exception as exc:
        print(f"gTTS failed: {exc}")
        TTS_DISABLED = True
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def say(text):
    output_encoding = sys.stdout.encoding or "utf-8"
    clean_text = clean_assistant_output(text)
    safe_text = clean_text.encode(output_encoding, errors="replace").decode(output_encoding)
    print(f"Jarvis: {safe_text}")
    speak(safe_text)


def recognize_audio(audio, recognizer):
    candidates = []
    for language in SPEECH_LANGUAGES:
        try:
            result = recognizer.recognize_google(audio, language=language, show_all=True)
        except Exception:
            continue
        alternatives = result.get("alternative", []) if isinstance(result, dict) else []
        candidates.extend(item.get("transcript", "") for item in alternatives if item.get("transcript"))
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = normalize_text(candidate)
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates


def listen_once(recognizer, source, choose_best_candidate):
    print("Listening...")
    audio = recognizer.listen(source, timeout=None, phrase_time_limit=LISTEN_PHRASE_SECONDS)
    candidates = recognize_audio(audio, recognizer)
    if candidates:
        print("I heard these possibilities:")
        for index, candidate in enumerate(candidates[:3], start=1):
            print(f"  {index}. {candidate}")
    text = choose_best_candidate(candidates)
    if text:
        print(f"You said: {text}")
    return text
