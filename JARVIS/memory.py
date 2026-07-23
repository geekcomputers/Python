import json

from .config import MAX_MEMORY_ITEMS, MEMORY_FILE


def load_json_list(path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def load_memory():
    return load_json_list(MEMORY_FILE)[-MAX_MEMORY_ITEMS:]


def save_memory(items):
    MEMORY_FILE.write_text(
        json.dumps(items[-MAX_MEMORY_ITEMS:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_memory():
    save_memory([])


def remember_note(note):
    note = str(note).strip()
    if not note:
        return "What should I remember?"
    memory = load_memory()
    if note not in memory:
        memory.append(note[:500])
    save_memory(memory)
    return "I will remember that."


def memory_context():
    memory = load_memory()
    if not memory:
        return "No saved memory yet."
    return "\n".join(f"- {item}" for item in memory[-10:])

