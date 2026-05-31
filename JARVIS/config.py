from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MEMORY_FILE = BASE_DIR / "jarvis_memory.json"
APP_INDEX_FILE = BASE_DIR / "jarvis_app_index.json"

OPENAI_API_KEY = "lm-studio"
OPENAI_BASE_URL = "http://localhost:1234/v1"
OPENAI_MODEL = "google/gemma-3-4b"
MAX_OUTPUT_TOKENS = 150
MAX_MEMORY_ITEMS = 30

SPEECH_LANGUAGES = ["en-US", "tr-TR"]
LISTEN_PHRASE_SECONDS = 12
TTS_MODE = "auto"

KNOWN_SITES = {
    "google": "https://www.google.com/",
    "youtube": "https://www.youtube.com/",
    "chatgpt": "https://chatgpt.com/",
    "chat gpt": "https://chatgpt.com/",
    "t3": "https://t3.chat/",
    "t3 chat": "https://t3.chat/",
    "github": "https://github.com/",
}

CMD_OPEN_PHRASES = {
    "cmd",
    "open cmd",
    "can you open cmd",
    "please open cmd",
    "cmd ac",
    "cmd aç",
    "command prompt",
    "open command prompt",
    "can you open command prompt",
    "please open command prompt",
    "komut istemi",
    "komut istemi ac",
    "komut istemi aç",
}

