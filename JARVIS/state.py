DEVELOPER_MODE = False


def set_developer_mode(enabled):
    global DEVELOPER_MODE
    DEVELOPER_MODE = bool(enabled)


def is_developer_mode():
    return DEVELOPER_MODE


def debug(label, value):
    if not DEVELOPER_MODE:
        return
    print(f"[dev] {label}: {value}")

